from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
import uuid

import numpy as np

from ml_inference.geo_projection import GeoContext, build_transformer, pixel_to_latlon


@dataclass
class PipelineModels:
    change_detector: Any | None = None
    aircraft_detector: Any | None = None
    classifier_fn: Callable[[np.ndarray], dict[str, Any]] | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _mask_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return (x1, y1, x2, y2)


def _event_base(
    event_type: str,
    lat: float,
    lon: float,
    confidence: float,
    bbox: list[float] | None,
    source: str,
    tile_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "event_type": event_type,
        "lat": float(lat),
        "lon": float(lon),
        "confidence": float(confidence),
        "bbox": bbox,
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "source": source,
        "tile_id": tile_id,
        "timestamp": _utc_now_iso(),
        "metadata": metadata or {},
    }


def _geo_for_pixel(
    x: float,
    y: float,
    geo_ctx: GeoContext | None,
    transformer: Any | None,
) -> tuple[float, float, bool]:
    if geo_ctx is None:
        return 0.0, 0.0, True
    lat, lon = pixel_to_latlon(x, y, geo_ctx, transformer=transformer)
    return lat, lon, False


def run_change_detection(
    before_img: np.ndarray,
    after_img: np.ndarray,
    geo_ctx: GeoContext | None,
    *,
    change_detector: Any,
    change_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformer = build_transformer(geo_ctx) if geo_ctx else None
    result = change_detector.run(before_img, after_img, semantic=False)
    mask = np.asarray(result.change_mask, dtype=np.float32)
    binary = (mask > float(change_threshold)).astype(np.uint8)
    changed_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    change_ratio = float(changed_pixels / max(1, total_pixels))
    bbox = _mask_bbox(binary)

    if bbox is not None:
        cx, cy = _bbox_center(bbox)
        lat, lon, missing = _geo_for_pixel(cx, cy, geo_ctx, transformer)
    else:
        lat, lon, missing = _geo_for_pixel(0.0, 0.0, geo_ctx, transformer)

    meta = {
        "change_ratio": change_ratio,
        "changed_pixels": changed_pixels,
        "threshold": float(change_threshold),
        "model_name": getattr(change_detector, "model_name", None),
        "device_used": getattr(change_detector, "runtime_device", None),
    }
    if missing:
        meta["geo_bounds_missing"] = True

    event = _event_base(
        event_type="CHANGE_DETECTED",
        lat=lat,
        lon=lon,
        confidence=min(1.0, max(0.0, change_ratio)),
        bbox=[float(v) for v in bbox] if bbox else None,
        source="change_detector",
        tile_id=geo_ctx.tile_id if geo_ctx else None,
        metadata=meta,
    )
    summary = {
        "change_ratio": change_ratio,
        "changed_pixels": changed_pixels,
    }
    return [event], summary


def run_aircraft_detection(
    image: np.ndarray,
    geo_ctx: GeoContext | None,
    *,
    aircraft_detector: Any,
    classifier_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
    max_detections: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transformer = build_transformer(geo_ctx) if geo_ctx else None
    detections = aircraft_detector.detect(image)
    if max_detections > 0:
        detections = detections[: max_detections]

    events: list[dict[str, Any]] = []
    for det in detections:
        bbox = tuple(det.bbox)
        cx, cy = _bbox_center(bbox)
        lat, lon, missing = _geo_for_pixel(cx, cy, geo_ctx, transformer)

        cls_payload: dict[str, Any] = {}
        if classifier_fn is not None:
            try:
                h, w = image.shape[:2]
                x1 = max(0, min(w - 1, int(round(bbox[0]))))
                y1 = max(0, min(h - 1, int(round(bbox[1]))))
                x2 = max(0, min(w, int(round(bbox[2]))))
                y2 = max(0, min(h, int(round(bbox[3]))))
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    cls_payload = classifier_fn(crop)
            except Exception:
                cls_payload = {}

        meta = {
            "class_id": cls_payload.get("class_id", None),
            "class_name": cls_payload.get("class_name", "Unknown"),
            "classifier_confidence": cls_payload.get("confidence", 0.0),
            "origin_country": cls_payload.get("origin_country", "Unknown"),
            "detector_model": getattr(aircraft_detector, "model_name", None),
        }
        if missing:
            meta["geo_bounds_missing"] = True

        events.append(
            _event_base(
                event_type="AIRCRAFT_DETECTED",
                lat=lat,
                lon=lon,
                confidence=float(det.confidence),
                bbox=[float(v) for v in bbox],
                source="aircraft_detector",
                tile_id=geo_ctx.tile_id if geo_ctx else None,
                metadata=meta,
            )
        )

    summary = {"aircraft_detections": len(detections)}
    return events, summary


def run_intelligence(
    before_img: np.ndarray | None,
    after_img: np.ndarray | None,
    geo_ctx: GeoContext | None,
    *,
    models: PipelineModels,
    run_change_detection: bool = True,
    run_aircraft_detection: bool = True,
    max_detections: int = 25,
    change_threshold: float = 0.5,
) -> dict[str, Any]:
    start = perf_counter()
    if before_img is None and after_img is None:
        raise ValueError("At least one image is required.")

    events: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    if run_change_detection:
        if before_img is None or after_img is None:
            raise ValueError("Change detection requires both before and after images.")
        if models.change_detector is None:
            raise ValueError("change_detector is required for change detection.")
        change_events, change_summary = run_change_detection(
            before_img,
            after_img,
            geo_ctx,
            change_detector=models.change_detector,
            change_threshold=change_threshold,
        )
        events.extend(change_events)
        summary.update(change_summary)

    if run_aircraft_detection:
        if after_img is None:
            raise ValueError("Aircraft detection requires an after image.")
        if models.aircraft_detector is None:
            raise ValueError("aircraft_detector is required for aircraft detection.")
        aircraft_events, aircraft_summary = run_aircraft_detection(
            after_img,
            geo_ctx,
            aircraft_detector=models.aircraft_detector,
            classifier_fn=models.classifier_fn,
            max_detections=max_detections,
        )
        events.extend(aircraft_events)
        summary.update(aircraft_summary)

    runtime_ms = (perf_counter() - start) * 1000.0
    processing = {
        "change_model": getattr(models.change_detector, "model_name", None),
        "aircraft_model": getattr(models.aircraft_detector, "model_name", None),
        "runtime_ms": float(runtime_ms),
        "tile_id": geo_ctx.tile_id if geo_ctx else None,
        "event_count": len(events),
        "timestamp": _utc_now_iso(),
    }
    return {"events": events, "summary": summary, "processing": processing}
