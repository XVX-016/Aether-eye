from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from aether_ml import AircraftDetectionPipeline, ChangeDetectionOnnxPipeline

from app.core.config import get_settings


@lru_cache
def get_aircraft_detector() -> AircraftDetectionPipeline:
    settings = get_settings()
    if not settings.aircraft_detector_onnx_path:
        raise RuntimeError("AIRCRAFT_DETECTOR_ONNX_PATH is not configured.")

    return AircraftDetectionPipeline(
        model_path=Path(settings.aircraft_detector_onnx_path),
        confidence_threshold=settings.aircraft_conf_threshold,
        iou_threshold=settings.aircraft_iou_threshold,
        device="auto",
    )


@lru_cache
def get_change_detector() -> ChangeDetectionOnnxPipeline:
    settings = get_settings()
    if not settings.change_detector_onnx_path:
        raise RuntimeError("CHANGE_DETECTOR_ONNX_PATH is not configured.")

    return ChangeDetectionOnnxPipeline(
        model_path=Path(settings.change_detector_onnx_path),
        device="auto",
    )

