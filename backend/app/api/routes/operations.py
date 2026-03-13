from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ActivityAlert, ObjectEvent, SatelliteScene, TileDetection
from app.database.session import get_db
from app.schemas.operations import CountResponse, OperationsEvent

router = APIRouter(tags=["operations"])


def _parse_bound(value: str, *, end_of_day: bool) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed_date = date.fromisoformat(normalized)
        parsed = datetime.combine(
            parsed_date,
            time.max if end_of_day else time.min,
            tzinfo=timezone.utc,
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@router.get("/events", response_model=list[OperationsEvent])
async def get_operations_events(
    hours: int = Query(default=24, ge=1, le=24 * 31),
    limit: int = Query(default=100, ge=1, le=500),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> list[OperationsEvent]:
    alert_query = (
        select(ActivityAlert, SatelliteScene.aoi_name)
        .outerjoin(
            SatelliteScene,
            SatelliteScene.aoi_name == ActivityAlert.location_name,
        )
        .order_by(ActivityAlert.triggered_at.desc())
    )
    object_query = (
        select(ObjectEvent, SatelliteScene.aoi_name)
        .outerjoin(
            SatelliteScene,
            SatelliteScene.scene_id == ObjectEvent.scene_id,
        )
        .order_by(ObjectEvent.timestamp.desc())
    )

    if start or end:
        if start:
            start_dt = _parse_bound(start, end_of_day=False)
            alert_query = alert_query.where(ActivityAlert.triggered_at >= start_dt)
            object_query = object_query.where(ObjectEvent.timestamp >= start_dt)
        if end:
            end_dt = _parse_bound(end, end_of_day=True)
            alert_query = alert_query.where(ActivityAlert.triggered_at <= end_dt)
            object_query = object_query.where(ObjectEvent.timestamp <= end_dt)
    else:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        alert_query = alert_query.where(ActivityAlert.triggered_at >= since)
        object_query = object_query.where(ObjectEvent.timestamp >= since)

    alert_rows = (await db.execute(alert_query.limit(limit))).all()
    if alert_rows:
        return [
            OperationsEvent(
                event_id=event.alert_id,
                event_type=event.alert_type,
                lat=event.lat,
                lon=event.lon,
                confidence=(event.payload or {}).get("confidence") if isinstance(event.payload, dict) else None,
                priority=event.severity,
                timestamp=event.triggered_at,
                aoi_name=event.location_name or aoi_name,
            )
            for event, aoi_name in alert_rows
            if event.lat is not None and event.lon is not None
        ]

    object_rows = (await db.execute(object_query.limit(limit))).all()
    return [
        OperationsEvent(
            event_id=event.event_id,
            event_type=event.type,
            lat=event.lat,
            lon=event.lon,
            confidence=event.confidence,
            priority=event.priority,
            timestamp=event.timestamp,
            aoi_name=aoi_name,
        )
        for event, aoi_name in object_rows
        if event.lat is not None and event.lon is not None
    ]


@router.get("/scenes/count", response_model=CountResponse)
async def get_scenes_count(db: AsyncSession = Depends(get_db)) -> CountResponse:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    count = await db.scalar(
        select(func.count()).select_from(SatelliteScene).where(SatelliteScene.created_at >= since)
    )
    return CountResponse(count=int(count or 0))


@router.get("/detections/count", response_model=CountResponse)
async def get_detections_count(db: AsyncSession = Depends(get_db)) -> CountResponse:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    count = await db.scalar(
        select(func.count()).select_from(TileDetection).where(TileDetection.created_at >= since)
    )
    return CountResponse(count=int(count or 0))


@router.get("/alerts/count", response_model=CountResponse)
async def get_alerts_count(db: AsyncSession = Depends(get_db)) -> CountResponse:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    count = await db.scalar(
        select(func.count()).select_from(ActivityAlert).where(ActivityAlert.created_at >= since)
    )
    return CountResponse(count=int(count or 0))
