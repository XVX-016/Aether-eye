from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from geoalchemy2.elements import WKTElement
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import AOIRegistry, AoiDailyCount, ObjectEvent, SatelliteScene, TileDetection


def _point_wkt(lat: float, lon: float) -> WKTElement:
    return WKTElement(f"POINT({lon} {lat})", srid=4326)


def _polygon_wkt_from_bbox(bbox: list[float]) -> WKTElement:
    min_lon, min_lat, max_lon, max_lat = bbox
    return WKTElement(
        (
            "POLYGON(("
            f"{min_lon} {min_lat}, "
            f"{max_lon} {min_lat}, "
            f"{max_lon} {max_lat}, "
            f"{min_lon} {max_lat}, "
            f"{min_lon} {min_lat}"
            "))"
        ),
        srid=4326,
    )


async def upsert_aoi(
    session: AsyncSession,
    *,
    aoi_id: str,
    name: str,
    bbox: list[float],
    scan_frequency_hrs: int = 6,
    cloud_threshold: float = 20.0,
    enabled: bool = True,
) -> AOIRegistry:
    result = await session.execute(select(AOIRegistry).where(AOIRegistry.aoi_id == aoi_id))
    aoi = result.scalar_one_or_none()
    polygon = _polygon_wkt_from_bbox(bbox)
    if aoi is None:
        aoi = AOIRegistry(
            aoi_id=aoi_id,
            name=name,
            bbox=bbox,
            polygon=polygon,
            scan_frequency_hrs=scan_frequency_hrs,
            cloud_threshold=cloud_threshold,
            enabled=enabled,
        )
        session.add(aoi)
    else:
        aoi.name = name
        aoi.bbox = bbox
        aoi.polygon = polygon
        aoi.scan_frequency_hrs = scan_frequency_hrs
        aoi.cloud_threshold = cloud_threshold
        aoi.enabled = enabled
    await session.flush()
    return aoi


async def list_enabled_aois(session: AsyncSession) -> list[dict[str, Any]]:
    result = await session.execute(
        select(
            AOIRegistry.aoi_id,
            AOIRegistry.name,
            AOIRegistry.bbox,
            AOIRegistry.scan_frequency_hrs,
            AOIRegistry.cloud_threshold,
            AOIRegistry.enabled,
            func.ST_AsGeoJSON(AOIRegistry.polygon),
        ).where(AOIRegistry.enabled.is_(True))
    )
    rows = result.all()
    return [
        {
            "aoi_id": row[0],
            "name": row[1],
            "bbox": row[2],
            "scan_frequency_hrs": row[3],
            "cloud_threshold": row[4],
            "enabled": row[5],
            "polygon_geojson": json.loads(row[6]) if row[6] else None,
        }
        for row in rows
    ]


async def save_scene(
    session: AsyncSession,
    *,
    scene_id: str,
    source: str,
    collection: str,
    aoi_id: str | None,
    aoi_name: str | None,
    dt: datetime,
    bbox: list[float] | None,
    cloud_cover: float | None,
    asset_href: str | None,
    geotiff_path: str | None = None,
    status: str = "DISCOVERED",
    processed: bool = False,
) -> SatelliteScene:
    result = await session.execute(select(SatelliteScene).where(SatelliteScene.scene_id == scene_id))
    scene = result.scalar_one_or_none()
    footprint = _polygon_wkt_from_bbox(bbox) if bbox else None
    if scene is None:
        scene = SatelliteScene(
            scene_id=scene_id,
            source=source,
            collection=collection,
            aoi_id=aoi_id,
            aoi_name=aoi_name,
            datetime=dt,
            bbox=bbox,
            footprint=footprint,
            cloud_cover=cloud_cover,
            asset_href=asset_href,
            geotiff_path=geotiff_path,
            status=status,
            processed=processed,
        )
        session.add(scene)
    else:
        scene.source = source
        scene.collection = collection
        scene.aoi_id = aoi_id
        scene.aoi_name = aoi_name
        scene.datetime = dt
        scene.bbox = bbox
        scene.footprint = footprint
        scene.cloud_cover = cloud_cover
        scene.asset_href = asset_href
        scene.geotiff_path = geotiff_path or scene.geotiff_path
        scene.status = status
        scene.processed = processed
    await session.flush()
    return scene


async def update_scene_status(
    session: AsyncSession,
    scene_id: str,
    *,
    status: str,
    geotiff_path: str | None = None,
    processed: bool | None = None,
    processed_at: datetime | None = None,
) -> SatelliteScene | None:
    result = await session.execute(select(SatelliteScene).where(SatelliteScene.scene_id == scene_id))
    scene = result.scalar_one_or_none()
    if scene is None:
        return None
    scene.status = status
    if geotiff_path is not None:
        scene.geotiff_path = geotiff_path
    if processed is not None:
        scene.processed = processed
    if processed_at is not None:
        scene.processed_at = processed_at
    await session.flush()
    return scene


async def get_scene_by_id(session: AsyncSession, scene_id: str) -> SatelliteScene | None:
    result = await session.execute(select(SatelliteScene).where(SatelliteScene.scene_id == scene_id))
    return result.scalar_one_or_none()


async def get_latest_processed_scene_for_aoi(
    session: AsyncSession,
    *,
    aoi_id: str | None,
    before_datetime: datetime,
    exclude_scene_id: str | None = None,
) -> SatelliteScene | None:
    query = select(SatelliteScene).where(
        SatelliteScene.processed.is_(True),
        SatelliteScene.datetime < before_datetime,
    )
    if aoi_id:
        query = query.where(SatelliteScene.aoi_id == aoi_id)
    if exclude_scene_id:
        query = query.where(SatelliteScene.scene_id != exclude_scene_id)
    query = query.order_by(SatelliteScene.datetime.desc()).limit(1)
    result = await session.execute(query)
    return result.scalar_one_or_none()


async def save_detection(
    session: AsyncSession,
    *,
    scene_id: str,
    tile_x: int,
    tile_y: int,
    lat: float,
    lon: float,
    model_type: str,
    change_score: float | None,
    confidence: float | None,
    detection_class: str | None,
    bbox: dict[str, float] | list[float] | None,
    metadata_json: dict[str, Any] | None = None,
) -> TileDetection:
    detection = TileDetection(
        scene_id=scene_id,
        tile_x=tile_x,
        tile_y=tile_y,
        lat=lat,
        lon=lon,
        location=_point_wkt(lat, lon),
        model_type=model_type,
        change_score=change_score,
        confidence=confidence,
        detection_class=detection_class,
        bbox=bbox,
        metadata_json=metadata_json,
    )
    session.add(detection)
    await session.flush()
    return detection


async def save_event(
    session: AsyncSession,
    *,
    event_id: str,
    event_type: str,
    scene_id: str | None,
    lat: float,
    lon: float,
    confidence: float | None,
    priority: str = "MEDIUM",
    detection_class: str | None = None,
    metadata_json: dict[str, Any] | None = None,
) -> ObjectEvent:
    event = ObjectEvent(
        event_id=event_id,
        type=event_type,
        scene_id=scene_id,
        lat=lat,
        lon=lon,
        location=_point_wkt(lat, lon),
        confidence=confidence,
        priority=priority,
        detection_class=detection_class,
        metadata_json=metadata_json,
    )
    session.add(event)
    await session.flush()
    return event


async def get_recent_events(
    session: AsyncSession,
    aoi_name: str,
    hours: int = 24,
) -> list[ObjectEvent]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await session.execute(
        select(ObjectEvent)
        .join(SatelliteScene, SatelliteScene.scene_id == ObjectEvent.scene_id, isouter=True)
        .where(SatelliteScene.aoi_name == aoi_name)
        .where(ObjectEvent.timestamp >= since)
        .order_by(ObjectEvent.timestamp.desc())
    )
    return list(result.scalars().all())


async def get_detection_history_for_cell(
    session: AsyncSession,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    days: int = 7,
) -> list[TileDetection]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    result = await session.execute(
        select(TileDetection)
        .where(TileDetection.timestamp >= since)
        .where(TileDetection.lat >= lat_min)
        .where(TileDetection.lat < lat_max)
        .where(TileDetection.lon >= lon_min)
        .where(TileDetection.lon < lon_max)
    )
    return list(result.scalars().all())


async def increment_aoi_daily_count(
    db: AsyncSession,
    aoi_id: str,
    date: date,
    event_type: str,
    increment: int = 1,
) -> None:
    stmt = insert(AoiDailyCount).values(
        aoi_id=aoi_id,
        date=date,
        event_type=event_type,
        count=increment,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_aoi_daily_counts_aoi_date_event",
        set_={
            "count": AoiDailyCount.count + stmt.excluded.count,
            "updated_at": func.now(),
        },
    )
    await db.execute(stmt)
    await db.flush()


async def get_aoi_baseline(
    db: AsyncSession,
    aoi_id: str,
    event_type: str,
    lookback_days: int = 30,
) -> float:
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=lookback_days)

    result = await db.execute(
        select(
            func.count(AoiDailyCount.id),
            func.avg(AoiDailyCount.count),
        )
        .where(AoiDailyCount.aoi_id == aoi_id)
        .where(AoiDailyCount.event_type == event_type)
        .where(AoiDailyCount.date >= since)
        .where(AoiDailyCount.date < today)
    )
    sample_count, average = result.one()
    if (sample_count or 0) < 3:
        return 0.0
    return float(average or 0.0)


async def backfill_aoi_daily_counts(
    db: AsyncSession,
) -> int:
    total_rows = 0

    event_aoi_expr = func.coalesce(SatelliteScene.aoi_id, "default")
    event_date_expr = func.date(ObjectEvent.created_at)
    result = await db.execute(
        select(
            event_aoi_expr.label("aoi_id"),
            event_date_expr.label("event_date"),
            ObjectEvent.type,
            func.count(ObjectEvent.id),
        )
        .outerjoin(SatelliteScene, SatelliteScene.scene_id == ObjectEvent.scene_id)
        .group_by(
            event_aoi_expr,
            event_date_expr,
            ObjectEvent.type,
        )
    )
    event_rows = result.all()
    if event_rows:
        event_values = [
            {
                "aoi_id": aoi_id,
                "date": event_date,
                "event_type": event_type,
                "count": count,
            }
            for aoi_id, event_date, event_type, count in event_rows
        ]

        stmt = insert(AoiDailyCount).values(event_values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_aoi_daily_counts_aoi_date_event",
            set_={
                "count": stmt.excluded.count,
                "updated_at": func.now(),
            },
        )
        await db.execute(stmt)
        total_rows += len(event_values)

    detection_aoi_expr = func.coalesce(SatelliteScene.aoi_id, "default")
    detection_date_expr = func.date(TileDetection.created_at)
    detection_result = await db.execute(
        select(
            detection_aoi_expr.label("aoi_id"),
            detection_date_expr.label("event_date"),
            func.count(TileDetection.id),
        )
        .outerjoin(SatelliteScene, SatelliteScene.scene_id == TileDetection.scene_id)
        .group_by(
            detection_aoi_expr,
            detection_date_expr,
        )
    )
    detection_rows = detection_result.all()
    if detection_rows:
        detection_values = [
            {
                "aoi_id": aoi_id,
                "date": event_date,
                "event_type": "detection",
                "count": count,
            }
            for aoi_id, event_date, count in detection_rows
        ]

        stmt = insert(AoiDailyCount).values(detection_values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_aoi_daily_counts_aoi_date_event",
            set_={
                "count": stmt.excluded.count,
                "updated_at": func.now(),
            },
        )
        await db.execute(stmt)
        total_rows += len(detection_values)

    await db.flush()
    return total_rows
