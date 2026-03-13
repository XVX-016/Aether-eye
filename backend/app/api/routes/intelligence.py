from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from typing import List, Dict, Any
from datetime import datetime, timezone
from time import perf_counter
import json
import numpy as np
import cv2
import logging

from app.schemas.intelligence import IntelligenceEvent, IntelligenceProcessRequest, IntelligenceProcessResponse, ActivityEvent, SceneRecord
from app.core.tasks import create_job, get_job, list_jobs, process_satellite_intelligence_task, JobStatus
from app.services.intelligence_service import GeoBounds, process_intelligence_arrays, persist_events
from app.database.session import get_db
from app.database.models import ActivityAlert as DBAircraftActivity, ObjectEvent as DBEvent, SatelliteScene as DBSatelliteScene
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ml_inference.geo_projection import GeoContext, geo_context_from_bounds
from app.api.upload_utils import (
    read_upload_image,
    parse_geo_bounds,
    http_error,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_PIXELS,
)

router = APIRouter(prefix="/intelligence", tags=["intelligence"])
logger = logging.getLogger(__name__)

@router.post("/process", response_model=Dict[str, str])
async def start_intelligence_processing(
    background_tasks: BackgroundTasks,
    payload: IntelligenceProcessRequest,
):
    """
    Starts an asynchronous intelligence processing job.
    Returns a job_id for polling.
    """
    job_id = create_job()
    bounds = None
    if payload.geo_bounds and len(payload.geo_bounds) == 4:
        bounds = GeoBounds(
            min_lat=float(payload.geo_bounds[0]),
            min_lon=float(payload.geo_bounds[1]),
            max_lat=float(payload.geo_bounds[2]),
            max_lon=float(payload.geo_bounds[3]),
        )
    background_tasks.add_task(
        process_satellite_intelligence_task, 
        job_id, 
        payload.image_before_path, 
        payload.image_after_path,
        bounds,
        payload.run_change_detection,
        payload.run_aircraft_detection,
        payload.max_detections,
        payload.use_classifier_onnx,
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


@router.get("/jobs", response_model=List[JobStatus])
async def get_jobs():
    return list_jobs()

@router.get("/events", response_model=List[IntelligenceEvent])
async def get_intelligence_events(db: AsyncSession = Depends(get_db)):
    """
    Fetch all persistent intelligence events from the database.
    """
    result = await db.execute(select(DBEvent).order_by(DBEvent.timestamp.desc()))
    events = result.scalars().all()
    out: List[IntelligenceEvent] = []
    for ev in events:
        meta = ev.metadata_json or {}
        out.append(
            IntelligenceEvent(
                event_id=ev.event_id,
                type=ev.type,
                lat=ev.lat,
                lon=ev.lon,
                confidence=ev.confidence,
                priority=ev.priority,
                timestamp=ev.timestamp,
                bbox=meta.get("bbox"),
                source=meta.get("source"),
                tile_id=meta.get("tile_id"),
                metadata=meta,
                geometry={"type": "Point", "coordinates": [ev.lon, ev.lat]},
            )
        )
    return out


@router.get("/activity", response_model=List[ActivityEvent])
async def get_activity_events(
    tile_id: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    def _parse_dt(value: str) -> datetime:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)

    query = select(DBAircraftActivity)
    if tile_id:
        query = query.where(DBAircraftActivity.tile_id == tile_id)
    if from_ts:
        query = query.where(DBAircraftActivity.window_start >= _parse_dt(from_ts))
    if to_ts:
        query = query.where(DBAircraftActivity.window_end <= _parse_dt(to_ts))
    result = await db.execute(query.order_by(DBAircraftActivity.window_end.desc()))
    rows = result.scalars().all()
    out: List[ActivityEvent] = []
    for ev in rows:
        out.append(
            ActivityEvent(
                tile_id=ev.tile_id,
                event_type=ev.event_type,
                window_start=ev.window_start,
                window_end=ev.window_end,
                previous_count=ev.previous_count,
                current_count=ev.current_count,
                delta=ev.delta,
                created_at=ev.created_at,
            )
        )
    return out


@router.get("/scenes", response_model=List[SceneRecord])
async def get_scenes(
    status: str | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    query = select(DBSatelliteScene)
    if status:
        query = query.where(DBSatelliteScene.status == status)
    query = query.order_by(DBSatelliteScene.datetime.desc()).limit(limit)
    result = await db.execute(query)
    rows = result.scalars().all()
    out: List[SceneRecord] = []
    for row in rows:
        out.append(
            SceneRecord(
                scene_id=row.scene_id,
                collection=row.collection,
                datetime=row.datetime,
                status=row.status,
                cloud_cover=row.cloud_cover,
                local_path=row.local_path,
            )
        )
    return out


@router.post("/process-upload", response_model=IntelligenceProcessResponse)
async def process_upload(
    before_image: UploadFile | None = File(default=None),
    after_image: UploadFile | None = File(default=None),
    geo_bounds: str | None = Form(default=None, description="Optional [min_lat,min_lon,max_lat,max_lon]."),
    run_change_detection: bool = Form(default=True),
    run_aircraft_detection: bool = Form(default=True),
    max_detections: int = Form(default=25),
    use_classifier_onnx: bool = Form(default=False),
    db: AsyncSession = Depends(get_db),
) -> IntelligenceProcessResponse:
    if before_image is None and after_image is None:
        raise http_error(400, "MISSING_IMAGE", "At least one image is required.")

    before_img = None
    after_img = None
    geo_ctx = None
    size_bytes = 0
    resolution = None
    t0 = perf_counter()

    if before_image is not None:
        before_img, geo_ctx, sz = read_upload_image(before_image)
        size_bytes += sz
        resolution = before_img.shape[:2]
    if after_image is not None:
        after_img, after_geo, sz = read_upload_image(after_image)
        geo_ctx = after_geo or geo_ctx
        size_bytes += sz
        resolution = after_img.shape[:2] or resolution

    bounds = parse_geo_bounds(geo_bounds)
    if geo_ctx is None and bounds is not None and after_img is not None:
        geo_ctx = geo_context_from_bounds(
            width=after_img.shape[1],
            height=after_img.shape[0],
            bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        )

    result = process_intelligence_arrays(
        before_img,
        after_img,
        geo_ctx,
        run_change_detection=run_change_detection,
        run_aircraft_detection=run_aircraft_detection,
        max_detections=max_detections,
        use_classifier_onnx=use_classifier_onnx,
    )
    runtime_ms = (perf_counter() - t0) * 1000.0

    # Persist events
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
        db.add(db_event)
    await db.commit()

    # Build response schema
    events_out = []
    now_ts = datetime.now(timezone.utc)
    for ev in normalized:
        meta = ev.get("metadata_json") or {}
        events_out.append(
            IntelligenceEvent(
                event_id=ev["event_id"],
                type=ev["type"],
                lat=ev["lat"],
                lon=ev["lon"],
                confidence=ev["confidence"],
                priority=ev["priority"],
                timestamp=now_ts,
                bbox=meta.get("bbox"),
                source=meta.get("source"),
                tile_id=meta.get("tile_id"),
                metadata=meta,
                geometry={"type": "Point", "coordinates": [ev["lon"], ev["lat"]]},
            )
        )

    event_count = len(events_out)
    processing = result.get("processing", {})
    processing["event_count"] = event_count
    processing["runtime_ms"] = runtime_ms

    if resolution:
        logger.info(
            "intelligence_upload size_bytes=%s resolution=%sx%s runtime_ms=%.2f events=%s",
            size_bytes,
            resolution[1],
            resolution[0],
            runtime_ms,
            event_count,
        )
    else:
        logger.info(
            "intelligence_upload size_bytes=%s runtime_ms=%.2f events=%s",
            size_bytes,
            runtime_ms,
            event_count,
        )

    return IntelligenceProcessResponse(events=events_out, summary=result["summary"], processing=processing)
