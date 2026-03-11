from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ml_inference.geo_projection import GeoContext, geo_context_from_bounds, read_geotiff_with_context
from ml_inference.pipeline import PipelineModels, run_intelligence

from app.services.change_service import get_change_detector_config, get_change_detector_v1
from app.services.onnx_model_service import get_aircraft_detector
from app.services.vit_service import classify_aircraft_onnx, get_vit_aircraft_pipeline


@dataclass
class GeoBounds:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image at path: {path}")
    return img


def _read_image_with_context(path: str | None) -> tuple[np.ndarray | None, GeoContext | None]:
    if not path:
        return None, None
    p = Path(path)
    if p.suffix.lower() in {".tif", ".tiff"}:
        return read_geotiff_with_context(p)
    return _read_image(str(p)), None


def _build_models(use_classifier_onnx: bool) -> PipelineModels:
    change_detector = get_change_detector_v1()
    aircraft_detector = get_aircraft_detector()
    if use_classifier_onnx:
        def _onnx_fn(img: np.ndarray) -> dict[str, Any]:
            res, _provider = classify_aircraft_onnx(img)
            return {
                "class_id": res.class_id,
                "class_name": res.class_name,
                "confidence": res.confidence,
                "origin_country": res.origin_country,
            }
        classifier_fn = _onnx_fn
    else:
        pipeline = get_vit_aircraft_pipeline()
        def _torch_fn(img: np.ndarray) -> dict[str, Any]:
            res = pipeline.classify(img)
            return {
                "class_id": res.class_id,
                "class_name": res.class_name,
                "confidence": res.confidence,
                "origin_country": res.origin_country,
            }
        classifier_fn = _torch_fn

    return PipelineModels(
        change_detector=change_detector,
        aircraft_detector=aircraft_detector,
        classifier_fn=classifier_fn,
    )


def process_intelligence_paths(
    image_before_path: str | None,
    image_after_path: str | None,
    *,
    geo_bounds: GeoBounds | None = None,
    run_change_detection: bool = True,
    run_aircraft_detection: bool = True,
    max_detections: int = 25,
    use_classifier_onnx: bool = False,
) -> dict[str, Any]:
    before_img, before_geo = _read_image_with_context(image_before_path)
    after_img, after_geo = _read_image_with_context(image_after_path)

    geo_ctx = after_geo or before_geo
    if geo_ctx is None and geo_bounds is not None and after_img is not None:
        geo_ctx = geo_context_from_bounds(
            width=after_img.shape[1],
            height=after_img.shape[0],
            bounds=(geo_bounds.min_lat, geo_bounds.min_lon, geo_bounds.max_lat, geo_bounds.max_lon),
        )

    models = _build_models(use_classifier_onnx)
    cfg = get_change_detector_config()
    return run_intelligence(
        before_img,
        after_img,
        geo_ctx,
        models=models,
        run_change_detection=run_change_detection,
        run_aircraft_detection=run_aircraft_detection,
        max_detections=max_detections,
        change_threshold=cfg.threshold,
    )


def process_intelligence_arrays(
    before_img: np.ndarray | None,
    after_img: np.ndarray | None,
    geo_ctx: GeoContext | None,
    *,
    run_change_detection: bool = True,
    run_aircraft_detection: bool = True,
    max_detections: int = 25,
    use_classifier_onnx: bool = False,
) -> dict[str, Any]:
    models = _build_models(use_classifier_onnx)
    cfg = get_change_detector_config()
    return run_intelligence(
        before_img,
        after_img,
        geo_ctx,
        models=models,
        run_change_detection=run_change_detection,
        run_aircraft_detection=run_aircraft_detection,
        max_detections=max_detections,
        change_threshold=cfg.threshold,
    )


def persist_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for ev in events:
        normalized.append(
            {
                "event_id": ev.get("event_id"),
                "type": ev.get("event_type"),
                "lat": ev.get("lat"),
                "lon": ev.get("lon"),
                "confidence": ev.get("confidence"),
                "priority": ev.get("priority", "MEDIUM"),
                "metadata_json": {
                    "bbox": ev.get("bbox"),
                    "source": ev.get("source"),
                    "tile_id": ev.get("tile_id"),
                    "timestamp": ev.get("timestamp"),
                    **(ev.get("metadata") or {}),
                },
            }
        )
    return normalized
