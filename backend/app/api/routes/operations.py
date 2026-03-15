from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.site_aggregator import get_site_status
from pipeline.site_registry import get_all_sites_geojson
from app.database.crud import get_aoi_baseline
from app.database.models import ActivityAlert, AoiDailyCount, ObjectEvent, SatelliteScene, TileDetection
from app.database.session import get_db
from app.schemas.operations import (
    AoiBaselineResponse,
    CountResponse,
    IntelArticleResponse,
    OperationsEvent,
    SiteStatusResponse,
)
from services.intel_feed import get_articles_for_site, get_global_articles

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


@router.get("/aoi-baseline", response_model=AoiBaselineResponse)
async def get_aoi_baseline_debug(
    aoi_id: str = Query(..., min_length=1),
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
) -> AoiBaselineResponse:
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=days)

    points_result = await db.execute(
        select(func.count(AoiDailyCount.id))
        .where(AoiDailyCount.aoi_id == aoi_id)
        .where(AoiDailyCount.date >= since)
        .where(AoiDailyCount.date < today)
    )
    data_points = int(points_result.scalar() or 0)

    event_types_result = await db.execute(
        select(AoiDailyCount.event_type)
        .where(AoiDailyCount.aoi_id == aoi_id)
        .where(AoiDailyCount.date >= since)
        .where(AoiDailyCount.date < today)
        .distinct()
    )
    event_types = {row[0] for row in event_types_result.fetchall()}
    event_types.update({"detection", "ACTIVITY_SURGE", "ELEVATED_ACTIVITY", "NEW_OBJECT"})

    baselines: dict[str, float] = {}
    for event_type in sorted(event_types):
        baselines[event_type] = await get_aoi_baseline(db, aoi_id, event_type, lookback_days=days)

    return AoiBaselineResponse(
        aoi_id=aoi_id,
        event_type_baselines=baselines,
        data_points=data_points,
        lookback_days=days,
    )


@router.get("/site-status", response_model=list[SiteStatusResponse])
async def get_site_status_endpoint(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
) -> list[SiteStatusResponse]:
    rows = await get_site_status(db, lookback_days=days)
    return [SiteStatusResponse(**row) for row in rows]


@router.get("/airbase-status", response_model=list[SiteStatusResponse], deprecated=True)
async def get_airbase_status_endpoint(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
) -> list[SiteStatusResponse]:
    rows = await get_site_status(db, lookback_days=days)
    return [SiteStatusResponse(**row) for row in rows]


@router.get("/sites/geojson")
async def get_sites_geojson() -> dict:
    return get_all_sites_geojson()


@router.get("/sites/{site_id}/intel", response_model=list[IntelArticleResponse])
async def get_site_intel(
    site_id: str,
    hours: int = Query(default=48, ge=1, le=24 * 30),
    db: AsyncSession = Depends(get_db),
) -> list[IntelArticleResponse]:
    rows = await get_articles_for_site(db, site_id, hours)
    return [IntelArticleResponse(site_id=site_id, **row) for row in rows]


@router.get("/intel/global", response_model=list[IntelArticleResponse])
async def get_global_intel(
    hours: int = Query(default=48, ge=1, le=24 * 30),
    db: AsyncSession = Depends(get_db),
) -> list[IntelArticleResponse]:
    rows = await get_global_articles(db, hours)
    return [IntelArticleResponse(**row) for row in rows]
