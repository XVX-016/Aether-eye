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
import uuid

async def process_satellite_intelligence_task(job_id: str, image_before_path: str, image_after_path: str):
    """
    Background task to process satellite intelligence.
    """
    try:
        update_job(job_id, status="processing", progress=0.1)
        
        # Real integration: 
        # 1. Call tiling engine + inference
        # 2. Extract events
        
        import time
        for i in range(2, 6):
            time.sleep(1) # Simulate tiling/inference
            update_job(job_id, progress=i/10)
            
        # Synthetic events for demonstration, mapped to spatial coordinates
        events_data = [
            {"event_id": f"evt_{uuid.uuid4().hex[:8]}", "type": "aircraft_arrival", "lat": 25.201, "lon": 55.269, "confidence": 0.93, "priority": "HIGH"},
            {"event_id": f"evt_{uuid.uuid4().hex[:8]}", "type": "construction_change", "lat": 25.210, "lon": 55.280, "confidence": 0.82, "priority": "MEDIUM"}
        ]
        
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
                session.add(db_event)
            await session.commit()
            
        update_job(job_id, status="completed", progress=1.0, result={"events_count": len(events_data)})
        
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
