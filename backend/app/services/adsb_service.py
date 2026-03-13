from __future__ import annotations

from typing import Any

import requests


OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"


def get_aircraft_in_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> list[dict[str, Any]]:
    response = requests.get(
        OPENSKY_STATES_URL,
        params={
            "lamin": lat_min,
            "lamax": lat_max,
            "lomin": lon_min,
            "lomax": lon_max,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json() or {}
    states = payload.get("states") or []
    out: list[dict[str, Any]] = []
    for state in states:
        if not state or len(state) < 17:
            continue
        out.append(
            {
                "icao24": state[0],
                "callsign": (state[1] or "").strip() or None,
                "origin_country": state[2],
                "lon": state[5],
                "lat": state[6],
                "altitude": state[7],
                "on_ground": state[8],
                "velocity": state[9],
                "heading": state[10],
            }
        )
    return out
