from __future__ import annotations

from typing import Any

from pipeline.site_registry import get_site_for_point, get_sites_by_type, load_sites

# DEPRECATED: kept as a thin compatibility shim while callers move to site_registry.py.


def load_airbases() -> list[dict[str, Any]]:
    return [site for site in load_sites() if "AIRBASE" in str(site.get("type", ""))]


def get_airbase_for_point(lat: float, lon: float) -> dict[str, Any] | None:
    site = get_site_for_point(lat, lon)
    if site is None:
        return None
    return site if "AIRBASE" in str(site.get("type", "")) else None


def get_airbases_for_bbox(bbox: list[float]) -> list[dict[str, Any]]:
    if len(bbox) != 4:
        return []
    min_lon, min_lat, max_lon, max_lat = bbox
    matches: list[dict[str, Any]] = []
    for airbase in get_sites_by_type("MILITARY_AIRBASE") + get_sites_by_type("CIVIL_AIRPORT"):
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
