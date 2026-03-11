import uuid
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json
import cv2
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
from app.database.models import IntelligenceEvent as DBEvent, SatelliteScene
from app.services.intelligence_service import GeoBounds, process_intelligence_paths, persist_events, process_intelligence_arrays
from app.services.ingestion_service import load_stac_config
from app.core.config import get_settings
from satellite_ingestion.stac_query import create_retrying_session
from ml_inference.geo_projection import geo_context_from_sidecar
from tiling_engine.tile_generator import TileGenerator
from sqlalchemy import select

async def process_satellite_intelligence_task(
    job_id: str,
    image_before_path: str | None,
    image_after_path: str | None,
    geo_bounds: GeoBounds | None = None,
    run_change_detection: bool = True,
    run_aircraft_detection: bool = True,
    max_detections: int = 25,
    use_classifier_onnx: bool = False,
):
    """
    Background task to process satellite intelligence.
    """
    try:
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
    Process a STAC scene: download if needed, tile, run aircraft detection, persist events.
    """
    if job_id is None:
        job_id = create_scene_job(scene_id)
    try:
        update_job(job_id, status="processing", progress=0.1, scene_id=scene_id)
        settings = get_settings()
        cfg = load_stac_config(settings)
        download_dir = Path(cfg.get("download_dir", "data/sentinel2_raw"))
        download_dir.mkdir(parents=True, exist_ok=True)

        async with async_session() as session:
            result = await session.execute(
                select(SatelliteScene).where(SatelliteScene.scene_id == scene_id)
            )
            scene = result.scalar_one_or_none()
            if scene is None:
                update_job(job_id, status="failed", error="Scene not found")
                return

            if not scene.local_path:
                local_path = download_dir / f"{scene.scene_id}.tif"
                session_retry = create_retrying_session()
                resp = session_retry.get(scene.asset_href, stream=True)
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                scene.local_path = str(local_path)
                scene.status = "DOWNLOADED"
                await session.commit()

            scene_path = Path(scene.local_path)
            tiles_dir = Path("data/tiles") / scene.scene_id
            tiler = TileGenerator(tile_size=512)
            tile_paths = tiler.generate_tiles(scene_path, tiles_dir)

            total = max(1, len(tile_paths))
            events_total = 0
            for idx, tile_path in enumerate(tile_paths):
                sidecar_path = tile_path.with_suffix(".json")
                if not sidecar_path.exists():
                    continue
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                geo_ctx = geo_context_from_sidecar(sidecar)
                img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                result = process_intelligence_arrays(
                    None,
                    img,
                    geo_ctx,
                    run_change_detection=False,
                    run_aircraft_detection=True,
                )
                normalized = persist_events(result["events"])
                for ev in normalized:
                    db_event = DBEvent(
                        event_id=ev["event_id"],
                        type=ev["type"],
                        lat=ev["lat"],
                        lon=ev["lon"],
                        confidence=ev["confidence"],
                        priority=ev["priority"],
                        metadata_json=ev.get("metadata_json"),
                    )
                    session.add(db_event)
                events_total += len(normalized)
                update_job(job_id, progress=(idx + 1) / total)

            scene.status = "PROCESSED"
            await session.commit()

        update_job(
            job_id,
            status="completed",
            progress=1.0,
            result={"events_count": events_total},
        )
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
