import uuid
from typing import Dict, Optional, List
from datetime import datetime
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

_jobs: Dict[str, JobStatus] = {}

def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="pending")
    return job_id

def update_job(job_id: str, **kwargs):
    if job_id in _jobs:
        job = _jobs[job_id]
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = datetime.now()

def get_job(job_id: str) -> Optional[JobStatus]:
    return _jobs.get(job_id)

from app.database.session import async_session
from app.database.models import IntelligenceEvent as DBEvent
from app.services.intelligence_service import GeoBounds, process_intelligence_paths, persist_events

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
