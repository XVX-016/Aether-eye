from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from app.database.crud import get_detection_history_for_cell, save_event
from app.database.session import async_session


GRID_SIZE_DEGREES = 0.01
MIN_EVENT_CHANGE_SCORE = 0.05
MIN_EVENT_CHANGED_PIXELS = 50


def _cell_bounds(lat: float, lon: float, grid_size: float = GRID_SIZE_DEGREES) -> tuple[float, float, float, float]:
    lat_bin = int(lat / grid_size)
    lon_bin = int(lon / grid_size)
    lat_min = lat_bin * grid_size
    lon_min = lon_bin * grid_size
    return lat_min, lat_min + grid_size, lon_min, lon_min + grid_size


async def generate_events(detections: list[dict[str, Any]], scene_id: str, persist: bool = True) -> list[dict[str, Any]]:
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
    events: list[dict[str, Any]] = []

    async with async_session() as session:
        for (_lat_bin, _lon_bin), cluster in clusters.items():
            current_count = len(cluster)
            mean_lat = sum(float(d["lat"]) for d in cluster) / current_count
            mean_lon = sum(float(d["lon"]) for d in cluster) / current_count
            lat_min, lat_max, lon_min, lon_max = _cell_bounds(mean_lat, mean_lon)
            history = await get_detection_history_for_cell(
                session,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                days=7,
            )
            historical_avg = len(history) / 7.0
            event_type: str | None = None
            if historical_avg == 0.0:
                event_type = "NEW_OBJECT"
            elif current_count / historical_avg > 3.0:
                event_type = "ACTIVITY_SURGE"
            if event_type is None:
                continue

            confidence = sum(float(d.get("confidence") or d.get("change_score") or 0.0) for d in cluster) / current_count
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "scene_id": scene_id,
                "lat": mean_lat,
                "lon": mean_lon,
                "confidence": confidence,
                "priority": "HIGH" if event_type == "ACTIVITY_SURGE" else "MEDIUM",
                "detection_class": cluster[0].get("detection_class"),
                "metadata_json": {
                    "cluster_size": current_count,
                    "historical_avg": historical_avg,
                    "surge_factor": (current_count / historical_avg) if historical_avg > 0 else None,
                    "grid_cell": [lat_min, lon_min, lat_max, lon_max],
                    "timestamp": now.isoformat(),
                },
            }
            events.append(event)
            if persist:
                await save_event(
                    session,
                    event_id=event["event_id"],
                    event_type=event["event_type"],
                    scene_id=scene_id,
                    lat=mean_lat,
                    lon=mean_lon,
                    confidence=confidence,
                    priority=event["priority"],
                    detection_class=event["detection_class"],
                    metadata_json=event["metadata_json"],
                )
        if persist:
            await session.commit()
    return events
