from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import AircraftActivityEvent, IntelligenceEvent as DBEvent


@dataclass
class ActivityEventPayload:
    tile_id: str
    event_type: str
    window_start: datetime
    window_end: datetime
    previous_count: int
    current_count: int
    delta: int


def compute_activity_events(
    events: Iterable[DBEvent],
    *,
    window_start: datetime,
    window_end: datetime,
    previous_window_start: datetime,
    previous_window_end: datetime,
    surge_factor: float,
    min_count: int,
) -> list[ActivityEventPayload]:
    current_counts: dict[str, int] = defaultdict(int)
    previous_counts: dict[str, int] = defaultdict(int)

    for ev in events:
        if not ev.metadata_json:
            continue
        tile_id = ev.metadata_json.get("tile_id")
        if not tile_id:
            continue
        ts = ev.timestamp
        if previous_window_start <= ts < previous_window_end:
            previous_counts[tile_id] += 1
        if window_start <= ts <= window_end:
            current_counts[tile_id] += 1

    payloads: list[ActivityEventPayload] = []
    for tile_id, current_count in current_counts.items():
        previous_count = previous_counts.get(tile_id, 0)
        if current_count < min_count:
            continue
        if previous_count == 0:
            if current_count >= min_count:
                payloads.append(
                    ActivityEventPayload(
                        tile_id=tile_id,
                        event_type="AIRCRAFT_SURGE",
                        window_start=window_start,
                        window_end=window_end,
                        previous_count=previous_count,
                        current_count=current_count,
                        delta=current_count - previous_count,
                    )
                )
            continue
        if current_count >= previous_count * surge_factor:
            payloads.append(
                ActivityEventPayload(
                    tile_id=tile_id,
                    event_type="AIRCRAFT_SURGE",
                    window_start=window_start,
                    window_end=window_end,
                    previous_count=previous_count,
                    current_count=current_count,
                    delta=current_count - previous_count,
                )
            )
    return payloads


async def aggregate_aircraft_activity(
    session: AsyncSession,
    *,
    window_hours: int,
    surge_factor: float,
    min_count: int,
) -> int:
    now = datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(hours=window_hours)
    previous_window_end = window_start
    previous_window_start = window_start - timedelta(hours=window_hours)

    result = await session.execute(
        select(DBEvent).where(DBEvent.type == "AIRCRAFT_DETECTED")
        .where(DBEvent.timestamp >= previous_window_start)
    )
    events = result.scalars().all()
    payloads = compute_activity_events(
        events,
        window_start=window_start,
        window_end=window_end,
        previous_window_start=previous_window_start,
        previous_window_end=previous_window_end,
        surge_factor=surge_factor,
        min_count=min_count,
    )

    created = 0
    for payload in payloads:
        exists = await session.execute(
            select(AircraftActivityEvent).where(
                AircraftActivityEvent.tile_id == payload.tile_id,
                AircraftActivityEvent.event_type == payload.event_type,
                AircraftActivityEvent.window_end == payload.window_end,
            )
        )
        if exists.scalar_one_or_none():
            continue
        event = AircraftActivityEvent(
            tile_id=payload.tile_id,
            event_type=payload.event_type,
            window_start=payload.window_start,
            window_end=payload.window_end,
            previous_count=payload.previous_count,
            current_count=payload.current_count,
            delta=payload.delta,
        )
        session.add(event)
        created += 1
    return created

