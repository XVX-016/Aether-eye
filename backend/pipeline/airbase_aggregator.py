from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.crud import get_aoi_baseline, increment_aoi_daily_count
from app.database.models import AoiDailyCount
from pipeline.airbase_monitor import get_airbase_for_point, load_airbases

logger = logging.getLogger(__name__)


async def aggregate_scene_for_airbases(
    scene_id: str,
    detections: list[dict[str, Any]],
    db: AsyncSession,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    airbase_lookup: dict[str, dict[str, Any]] = {}

    for detection in detections:
        lat = detection.get("lat")
        lon = detection.get("lon")
        if lat is None or lon is None:
            continue
        airbase = get_airbase_for_point(float(lat), float(lon))
        if airbase is None:
            continue
        airbase_id = str(airbase["id"])
        counts[airbase_id] += 1
        airbase_lookup[airbase_id] = airbase

    today = datetime.now(timezone.utc).date()
    for airbase_id, count in counts.items():
        if count <= 0:
            continue
        airbase = airbase_lookup[airbase_id]
        await increment_aoi_daily_count(db, airbase_id, today, "detection", count)
        baseline = await get_aoi_baseline(db, airbase_id, "detection", lookback_days=30)
        anomaly_factor = (count / baseline) if baseline > 0 else None
        if (anomaly_factor is not None and anomaly_factor >= 2.0) or (baseline == 0 and count > 5):
            factor_text = f"{anomaly_factor:.2f}" if anomaly_factor is not None else "None"
            logger.warning(
                "AIRBASE ANOMALY: %s detections=%s baseline=%.1f factor=%s scene_id=%s",
                airbase.get("name", airbase_id),
                count,
                baseline,
                factor_text,
                scene_id,
            )

    return dict(counts)


async def get_airbase_status(
    db: AsyncSession,
    lookback_days: int = 30,
) -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    statuses: list[dict[str, Any]] = []

    for airbase in load_airbases():
        airbase_id = str(airbase["id"])
        baseline = await get_aoi_baseline(db, airbase_id, "detection", lookback_days=lookback_days)
        today_result = await db.execute(
            select(AoiDailyCount.count)
            .where(AoiDailyCount.aoi_id == airbase_id)
            .where(AoiDailyCount.event_type == "detection")
            .where(AoiDailyCount.date == today)
            .order_by(desc(AoiDailyCount.updated_at))
            .limit(1)
        )
        today_count = int(today_result.scalar() or 0)
        anomaly_factor = (today_count / baseline) if baseline > 0 else None

        if anomaly_factor is not None and anomaly_factor >= 2.0:
            status = "anomalous"
        elif anomaly_factor is not None and anomaly_factor >= 1.5:
            status = "elevated"
        else:
            status = "normal"

        statuses.append(
            {
                "id": airbase_id,
                "name": airbase.get("name"),
                "type": airbase.get("type"),
                "priority": airbase.get("priority"),
                "country": airbase.get("country"),
                "today_count": today_count,
                "baseline": baseline,
                "anomaly_factor": anomaly_factor,
                "status": status,
            }
        )

    return statuses
