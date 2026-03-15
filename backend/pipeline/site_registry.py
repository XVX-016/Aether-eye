from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "monitoring" / "global_sites.yaml"


@lru_cache(maxsize=1)
def load_sites() -> list[dict[str, Any]]:
    path = _config_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return list(payload.get("sites", []))


def get_site_for_point(lat: float, lon: float) -> dict[str, Any] | None:
    for site in load_sites():
        bbox = site.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        min_lon, min_lat, max_lon, max_lat = bbox
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return site
    return None


def get_sites_by_type(site_type: str) -> list[dict[str, Any]]:
    return [site for site in load_sites() if str(site.get("type")) == site_type]


def get_sites_by_priority(priority: str) -> list[dict[str, Any]]:
    return [site for site in load_sites() if str(site.get("priority")) == priority]


def get_all_sites_geojson() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(site["lon"]), float(site["lat"])],
                },
                "properties": dict(site),
            }
            for site in load_sites()
            if site.get("lat") is not None and site.get("lon") is not None
        ],
    }
