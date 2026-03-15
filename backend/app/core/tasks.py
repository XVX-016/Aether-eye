import uuid
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import logging
from pydantic import BaseModel, Field

# In-memory job store for prototype (transition to Redis/DB later)
class JobStatus(BaseModel):
    job_id: str
    status: str # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    scene_id: Optional[str] = None

_jobs: Dict[str, JobStatus] = {}
logger = logging.getLogger(__name__)

def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="pending")
    return job_id


def create_scene_job(scene_id: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="pending", scene_id=scene_id)
    return job_id

def update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        job = _jobs[job_id]
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = datetime.now()

def get_job(job_id: str) -> Optional[JobStatus]:
    return _jobs.get(job_id)


def list_jobs() -> List[JobStatus]:
    return list(_jobs.values())

from app.database.session import async_session
from app.database.models import ObjectEvent as DBEvent
from app.database.crud import get_latest_processed_scene_for_aoi, get_scene_by_id, update_scene_status
from app.services.ingestion_service import load_stac_config
from app.core.config import get_settings
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_retrying_session() -> Session:
    session = Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

async def process_satellite_intelligence_task(
    job_id: str,
    image_before_path: str | None,
    image_after_path: str | None,
    geo_bounds = None,
    run_change_detection: bool = True,
    run_aircraft_detection: bool = True,
    max_detections: int = 25,
    use_classifier_onnx: bool = False,
):
    """
    Background task to process satellite intelligence.
    """
    try:
        from app.services.intelligence_service import process_intelligence_paths, persist_events

        update_job(job_id, status="processing", progress=0.1)

        result = process_intelligence_paths(
            image_before_path,
            image_after_path,
            geo_bounds=geo_bounds,
            run_change_detection=run_change_detection,
            run_aircraft_detection=run_aircraft_detection,
            max_detections=max_detections,
            use_classifier_onnx=use_classifier_onnx,
        )
        events_data = persist_events(result["events"])

        # Save to DB
        async with async_session() as session:
            for ev in events_data:
                db_event = DBEvent(
                    event_id=ev["event_id"],
                    type=ev["type"],
                    lat=ev["lat"],
                    lon=ev["lon"],
                    confidence=ev["confidence"],
                    priority=ev["priority"]
                )
                if "metadata_json" in ev:
                    db_event.metadata_json = ev["metadata_json"]
                session.add(db_event)
            await session.commit()
            
        update_job(
            job_id,
            status="completed",
            progress=1.0,
            result={
                "events_count": len(events_data),
                "summary": result.get("summary", {}),
            },
        )
        
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))


async def process_scene_job(scene_id: str, job_id: str | None = None) -> None:
    """
    Process a STAC scene via the Stage 2 change-detection pipeline.
    """
    if job_id is None:
        job_id = create_scene_job(scene_id)
    try:
        from pipeline.site_aggregator import aggregate_scene_for_sites
        from pipeline.event_engine import generate_events
        from pipeline.scene_processor import process_scene

        update_job(job_id, status="processing", progress=0.1, scene_id=scene_id)
        settings = get_settings()
        cfg = load_stac_config(settings)
        download_dir = Path(cfg.get("download_dir", "data/sentinel2_raw"))
        download_dir.mkdir(parents=True, exist_ok=True)

        async with async_session() as session:
            scene = await get_scene_by_id(session, scene_id)
            if scene is None:
                update_job(job_id, status="failed", error="Scene not found")
                return

            if not scene.geotiff_path:
                local_path = download_dir / f"{scene.scene_id}.tif"
                session_retry = create_retrying_session()
                resp = session_retry.get(scene.asset_href, stream=True)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                await update_scene_status(
                    session,
                    scene.scene_id,
                    status="DOWNLOADED",
                    geotiff_path=str(local_path),
                )
                await session.commit()

            previous_scene = await get_latest_processed_scene_for_aoi(
                session,
                aoi_id=scene.aoi_id,
                before_datetime=scene.datetime,
                exclude_scene_id=scene.scene_id,
            )
            await session.commit()

        scene_path = Path(scene.geotiff_path or "")
        previous_path = previous_scene.geotiff_path if previous_scene is not None else None
        detections = await process_scene(
            str(scene_path),
            previous_path,
            scene_id=scene.scene_id,
            spectral_threshold=0.05,
            overlap=0,
            semantic=False,
        )
        async with async_session() as session:
            events = await generate_events(detections, scene.scene_id, session)
            try:
                await aggregate_scene_for_sites(scene.scene_id, detections, session)
                await session.commit()
            except Exception as agg_exc:
                await session.rollback()
                logger.warning("Site aggregation failed for scene %s: %s", scene.scene_id, agg_exc)

        update_job(
            job_id,
            status="completed",
            progress=1.0,
            result={"detections_count": len(detections), "events_count": len(events)},
        )
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
