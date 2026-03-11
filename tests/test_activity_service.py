from datetime import datetime, timedelta, timezone
import pytest

try:
    import aiosqlite  # noqa: F401
except Exception:
    pytest.skip("aiosqlite not available", allow_module_level=True)

from app.database.models import IntelligenceEvent as DBEvent
from app.services.activity_service import compute_activity_events


def test_compute_activity_events_surge():
    now = datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(hours=24)
    prev_end = window_start
    prev_start = prev_end - timedelta(hours=24)

    events = []
    # previous window: 2 aircraft
    for _ in range(2):
        events.append(
            DBEvent(
                type="AIRCRAFT_DETECTED",
                timestamp=prev_start + timedelta(hours=1),
                metadata_json={"tile_id": "tile_1"},
            )
        )
    # current window: 6 aircraft
    for _ in range(6):
        events.append(
            DBEvent(
                type="AIRCRAFT_DETECTED",
                timestamp=window_start + timedelta(hours=1),
                metadata_json={"tile_id": "tile_1"},
            )
        )

    payloads = compute_activity_events(
        events,
        window_start=window_start,
        window_end=window_end,
        previous_window_start=prev_start,
        previous_window_end=prev_end,
        surge_factor=2.0,
        min_count=3,
    )

    assert len(payloads) == 1
    assert payloads[0].tile_id == "tile_1"
    assert payloads[0].event_type == "AIRCRAFT_SURGE"
    assert payloads[0].previous_count == 2
    assert payloads[0].current_count == 6
