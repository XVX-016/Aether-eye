from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "monitoring" / "airbases.yaml"


@lru_cache(maxsize=1)
def load_airbases() -> list[dict[str, Any]]:
    path = _config_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return list(payload.get("airbases", []))


def get_airbase_for_point(lat: float, lon: float) -> dict[str, Any] | None:
    for airbase in load_airbases():
        bbox = airbase.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return airbase
    return None


def get_airbases_for_bbox(bbox: list[float]) -> list[dict[str, Any]]:
    if len(bbox) != 4:
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    matches: list[dict[str, Any]] = []
    for airbase in load_airbases():
        airbase_bbox = airbase.get("bbox")
        if not airbase_bbox or len(airbase_bbox) != 4:
            continue
        a_min_lon, a_min_lat, a_max_lon, a_max_lat = airbase_bbox
        overlaps = not (
            a_max_lon < min_lon or
            a_min_lon > max_lon or
            a_max_lat < min_lat or
            a_min_lat > max_lat
        )
        if overlaps:
            matches.append(airbase)
    return matches
