from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_geojson(events: list[dict[str, Any]]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for ev in events:
        lat = ev.get("lat")
        lon = ev.get("lon")
        if lat is None or lon is None:
            continue
        props = {k: v for k, v in ev.items() if k not in {"lat", "lon"}}
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_events(events: list[dict[str, Any]], output_path: str | Path, fmt: str = "json") -> Path:
    output_path = Path(output_path)
    fmt_norm = fmt.strip().lower()
    if fmt_norm not in {"json", "geojson"}:
        raise ValueError("format must be 'json' or 'geojson'")

    payload = events if fmt_norm == "json" else _to_geojson(events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def write_result(result: dict[str, Any], output_path: str | Path, fmt: str = "json") -> Path:
    output_path = Path(output_path)
    fmt_norm = fmt.strip().lower()
    if fmt_norm not in {"json", "geojson"}:
        raise ValueError("format must be 'json' or 'geojson'")

    if fmt_norm == "json":
        payload: dict[str, Any] = {
            "events": result.get("events", []),
            "summary": result.get("summary", {}),
            "processing": result.get("processing", {}),
        }
    else:
        payload = _to_geojson(result.get("events", []))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
