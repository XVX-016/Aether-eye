from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime

# Assuming these are available or mocked for now
from app.schemas.intelligence import IntelligenceEvent, DetectionRequest
from app.core.tasks import create_job, get_job, process_satellite_intelligence_task, JobStatus

router = APIRouter(prefix="/intelligence", tags=["intelligence"])

@router.post("/process", response_model=Dict[str, str])
async def start_intelligence_processing(
    background_tasks: BackgroundTasks,
    image_before: str, # In production, these would be file uploads or S3 keys
    image_after: str
):
    """
    Starts an asynchronous intelligence processing job.
    Returns a job_id for polling.
    """
    job_id = create_job()
    background_tasks.add_task(
        process_satellite_intelligence_task, 
        job_id, 
        image_before, 
        image_after
    )
    return {"job_id": job_id}

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_processing_status(job_id: str):
    """
    Poll the status of an intelligence processing job.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import get_db
from app.database.models import IntelligenceEvent as DBEvent
from fastapi import Depends

@router.get("/events", response_model=List[IntelligenceEvent])
async def get_intelligence_events(db: AsyncSession = Depends(get_db)):
    """
    Fetch all persistent intelligence events from the database.
    """
    result = await db.execute(select(DBEvent).order_by(DBEvent.timestamp.desc()))
    events = result.scalars().all()
    
    # Convert DB models to Pydantic schemas if needed, or rely on FastAPI's auto-conversion
    return events
