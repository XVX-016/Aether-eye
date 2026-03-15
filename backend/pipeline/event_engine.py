from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.crud import get_aoi_baseline, increment_aoi_daily_count, save_event
from pipeline.site_registry import get_site_for_point


GRID_SIZE_DEGREES = 0.01
MIN_EVENT_CHANGE_SCORE = 0.05
MIN_EVENT_CHANGED_PIXELS = 50


def _cell_bounds(lat: float, lon: float, grid_size: float = GRID_SIZE_DEGREES) -> tuple[float, float, float, float]:
    lat_bin = int(lat / grid_size)
    lon_bin = int(lon / grid_size)
    lat_min = lat_bin * grid_size
    lon_min = lon_bin * grid_size
    return lat_min, lat_min + grid_size, lon_min, lon_min + grid_size


def _cell_center(cluster: list[dict[str, Any]]) -> tuple[float, float]:
    count = len(cluster)
    mean_lat = sum(float(item["lat"]) for item in cluster) / count
    mean_lon = sum(float(item["lon"]) for item in cluster) / count
    return mean_lat, mean_lon


def _resolve_aoi_id(lat: float, lon: float) -> str:
    site = get_site_for_point(lat, lon)
    return str(site.get("id")) if site is not None else "default"


async def generate_events(
    detections: list[dict[str, Any]],
    scene_id: str,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    clusters: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for det in detections:
        change_score = float(det.get("change_score") or 0.0)
        metadata = det.get("metadata_json") or {}
        changed_pixels = int(metadata.get("changed_pixels") or 0)
        if change_score <= MIN_EVENT_CHANGE_SCORE or changed_pixels <= MIN_EVENT_CHANGED_PIXELS:
            continue
        lat = float(det["lat"])
        lon = float(det["lon"])
        key = (int(lat / GRID_SIZE_DEGREES), int(lon / GRID_SIZE_DEGREES))
        clusters[key].append(det)

    now = datetime.now(timezone.utc)
    today = now.date()
    events: list[dict[str, Any]] = []
    detection_increments: dict[str, int] = defaultdict(int)
    event_type_increments: dict[tuple[str, str], int] = defaultdict(int)

    for cluster in clusters.values():
        current_count = len(cluster)
        mean_lat, mean_lon = _cell_center(cluster)
        lat_min, lat_max, lon_min, lon_max = _cell_bounds(mean_lat, mean_lon)
        aoi_id = _resolve_aoi_id(mean_lat, mean_lon)
        baseline = await get_aoi_baseline(db, aoi_id, "detection", lookback_days=30)
        surge_factor = (current_count / baseline) if baseline > 0 else None

        event_type: str | None = None
        if baseline == 0 and current_count > 0:
            event_type = "NEW_OBJECT"
        elif surge_factor is not None and surge_factor >= 3.0:
            event_type = "ACTIVITY_SURGE"
        elif surge_factor is not None and surge_factor >= 1.5:
            event_type = "ELEVATED_ACTIVITY"
        else:
            detection_increments[aoi_id] += current_count
            continue

        confidence = sum(float(d.get("confidence") or d.get("change_score") or 0.0) for d in cluster) / current_count
        priority = "HIGH" if event_type == "ACTIVITY_SURGE" else "MEDIUM" if event_type == "ELEVATED_ACTIVITY" else "LOW"
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "scene_id": scene_id,
            "lat": mean_lat,
            "lon": mean_lon,
            "confidence": confidence,
            "priority": priority,
            "detection_class": cluster[0].get("detection_class"),
            "metadata_json": {
                "cluster_size": current_count,
                "historical_avg": baseline,
                "surge_factor": surge_factor,
                "grid_cell": [lat_min, lon_min, lat_max, lon_max],
                "timestamp": now.isoformat(),
                "aoi_id": aoi_id,
            },
        }
        events.append(event)
        await save_event(
            db,
            event_id=event["event_id"],
            event_type=event["event_type"],
            scene_id=scene_id,
            lat=mean_lat,
            lon=mean_lon,
            confidence=confidence,
            priority=priority,
            detection_class=event["detection_class"],
            metadata_json=event["metadata_json"],
        )
        detection_increments[aoi_id] += current_count
        event_type_increments[(aoi_id, event_type)] += 1

    for aoi_id, increment in detection_increments.items():
        if increment > 0:
            await increment_aoi_daily_count(db, aoi_id, today, "detection", increment)

    for (aoi_id, event_type), increment in event_type_increments.items():
        await increment_aoi_daily_count(db, aoi_id, today, event_type, increment)

    await db.commit()
    return events
