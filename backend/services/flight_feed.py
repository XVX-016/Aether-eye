from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from app.database.models import FlightDailyCount, FlightState
    from pipeline.site_registry import get_site_for_point, load_sites
except ModuleNotFoundError:
    from backend.app.database.models import FlightDailyCount, FlightState
    from backend.pipeline.site_registry import get_site_for_point, load_sites


logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 300
SITE_BUFFER_DEGREES = 0.5
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"


def _expanded_bbox(bbox: list[float]) -> dict[str, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "lamin": min_lat - SITE_BUFFER_DEGREES,
        "lomin": min_lon - SITE_BUFFER_DEGREES,
        "lamax": max_lat + SITE_BUFFER_DEGREES,
        "lomax": max_lon + SITE_BUFFER_DEGREES,
    }


def _coerce_timestamp(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def _clean_callsign(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def _upsert_daily_count(db: AsyncSession, site_id: str, target_date) -> None:
    totals = await db.execute(
        select(
            func.count(FlightState.id),
            func.count(distinct(FlightState.icao24)),
        )
        .where(FlightState.site_id == site_id)
        .where(func.date(FlightState.timestamp) == target_date)
    )
    count, unique_aircraft = totals.one()

    stmt = insert(FlightDailyCount).values(
        site_id=site_id,
        date=target_date,
        count=int(count or 0),
        unique_aircraft=int(unique_aircraft or 0),
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_flight_daily_counts_site_date",
        set_={
            "count": stmt.excluded.count,
            "unique_aircraft": stmt.excluded.unique_aircraft,
            "updated_at": func.now(),
        },
    )
    await db.execute(stmt)


async def fetch_flights_for_sites(db: AsyncSession) -> int:
    sites = load_sites()
    total_states = 0
    today = datetime.now(timezone.utc).date()

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for site in sites:
            site_id = str(site["id"])
            try:
                expanded = _expanded_bbox(list(site["bbox"]))
                response = await client.get(OPENSKY_STATES_URL, params=expanded)
                response.raise_for_status()
                payload = response.json() or {}
                states = payload.get("states") or []

                rows_to_insert: list[dict[str, Any]] = []
                for state in states:
                    if not isinstance(state, list) or len(state) < 11:
                        continue

                    icao24 = str(state[0]).strip() if state[0] is not None else ""
                    if not icao24:
                        continue

                    lon = float(state[5]) if state[5] is not None else None
                    lat = float(state[6]) if state[6] is not None else None
                    assigned_site_id = None
                    if lat is not None and lon is not None:
                        matched = get_site_for_point(lat, lon)
                        assigned_site_id = str(matched["id"]) if matched else None

                    rows_to_insert.append(
                        {
                            "icao24": icao24,
                            "callsign": _clean_callsign(state[1]),
                            "origin_country": str(state[2]).strip() if state[2] is not None else None,
                            "lat": lat,
                            "lon": lon,
                            "altitude_m": float(state[7]) if state[7] is not None else None,
                            "velocity_ms": float(state[9]) if state[9] is not None else None,
                            "heading": float(state[10]) if state[10] is not None else None,
                            "on_ground": bool(state[8]) if state[8] is not None else False,
                            "site_id": assigned_site_id,
                            "timestamp": _coerce_timestamp(state[3] or payload.get("time")),
                        }
                    )

                if rows_to_insert:
                    stmt = insert(FlightState).values(rows_to_insert)
                    stmt = stmt.on_conflict_do_nothing(constraint="uq_flight_states_icao24_timestamp")
                    result = await db.execute(stmt)
                    total_states += int(result.rowcount or 0)

                await _upsert_daily_count(db, site_id, today)
                await db.commit()
            except Exception as exc:
                await db.rollback()
                logger.warning("Flight feed failed for %s: %s", site_id, exc)
                continue

    return total_states


async def get_flight_activity_for_site(
    db: AsyncSession,
    site_id: str,
    hours: int = 24,
) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    summary_result = await db.execute(
        select(
            func.count(FlightState.id),
            func.count(distinct(FlightState.icao24)),
            func.count().filter(FlightState.on_ground.is_(True)),
            func.count().filter(FlightState.on_ground.is_(False)),
        )
        .where(FlightState.site_id == site_id)
        .where(FlightState.timestamp >= since)
    )
    recent_count, unique_aircraft, on_ground_count, airborne_count = summary_result.one()

    latest_rows = (
        await db.execute(
            select(FlightState)
            .where(FlightState.site_id == site_id)
            .order_by(FlightState.timestamp.desc(), FlightState.id.desc())
            .limit(10)
        )
    ).scalars().all()

    return {
        "site_id": site_id,
        "recent_count": int(recent_count or 0),
        "unique_aircraft": int(unique_aircraft or 0),
        "on_ground_count": int(on_ground_count or 0),
        "airborne_count": int(airborne_count or 0),
        "latest_states": [
            {
                "icao24": row.icao24,
                "callsign": row.callsign,
                "origin_country": row.origin_country,
                "lat": row.lat,
                "lon": row.lon,
                "altitude_m": row.altitude_m,
                "velocity_ms": row.velocity_ms,
                "heading": row.heading,
                "on_ground": row.on_ground,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            }
            for row in latest_rows
        ],
    }


async def get_flight_baseline(
    db: AsyncSession,
    site_id: str,
    lookback_days: int = 30,
) -> float:
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=lookback_days)

    result = await db.execute(
        select(
            func.count(FlightDailyCount.id),
            func.avg(FlightDailyCount.unique_aircraft),
        )
        .where(FlightDailyCount.site_id == site_id)
        .where(FlightDailyCount.date >= since)
        .where(FlightDailyCount.date < today)
    )
    sample_count, average = result.one()
    if int(sample_count or 0) < 3:
        return 0.0
    return float(average or 0.0)
