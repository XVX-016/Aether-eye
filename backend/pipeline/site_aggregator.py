from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.crud import get_aoi_baseline, increment_aoi_daily_count
from app.database.models import AoiDailyCount
from pipeline.site_registry import get_site_for_point, load_sites

logger = logging.getLogger(__name__)


async def aggregate_scene_for_sites(
    scene_id: str,
    detections: list[dict[str, Any]],
    db: AsyncSession,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    site_lookup: dict[str, dict[str, Any]] = {}

    for detection in detections:
        lat = detection.get("lat")
        lon = detection.get("lon")
        if lat is None or lon is None:
            continue
        site = get_site_for_point(float(lat), float(lon))
        if site is None:
            continue
        site_id = str(site["id"])
        counts[site_id] += 1
        site_lookup[site_id] = site

    today = datetime.now(timezone.utc).date()
    for site_id, count in counts.items():
        if count <= 0:
            continue
        site = site_lookup[site_id]
        await increment_aoi_daily_count(db, site_id, today, "detection", count)
        baseline = await get_aoi_baseline(db, site_id, "detection", lookback_days=30)
        anomaly_factor = (count / baseline) if baseline > 0 else None
        if (anomaly_factor is not None and anomaly_factor >= 2.0) or (baseline == 0 and count > 5):
            factor_text = f"{anomaly_factor:.2f}" if anomaly_factor is not None else "None"
            logger.warning(
                "SITE ANOMALY: %s detections=%s baseline=%.1f factor=%s scene_id=%s",
                site.get("name", site_id),
                count,
                baseline,
                factor_text,
                scene_id,
            )

    return dict(counts)


async def get_site_status(
    db: AsyncSession,
    lookback_days: int = 30,
) -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date()
    statuses: list[dict[str, Any]] = []

    for site in load_sites():
        site_id = str(site["id"])
        baseline = await get_aoi_baseline(db, site_id, "detection", lookback_days=lookback_days)
        today_result = await db.execute(
            select(AoiDailyCount.count)
            .where(AoiDailyCount.aoi_id == site_id)
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
                "id": site_id,
                "name": site.get("name"),
                "type": site.get("type"),
                "priority": site.get("priority"),
                "country": site.get("country"),
                "today_count": today_count,
                "baseline": baseline,
                "anomaly_factor": anomaly_factor,
                "status": status,
            }
        )

    return statuses
