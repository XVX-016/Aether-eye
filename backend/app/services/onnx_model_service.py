from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from aether_ml import AircraftDetectionPipeline, ChangeDetectionOnnxPipeline

from app.core.config import get_settings


@lru_cache
def get_aircraft_detector() -> AircraftDetectionPipeline:
    if AircraftDetectionPipeline is None:
        raise RuntimeError("AircraftDetectionPipeline unavailable. Check ONNX runtime installation.")
    settings = get_settings()
    if not settings.aircraft_detector_onnx_path:
        raise RuntimeError("AIRCRAFT_DETECTOR_ONNX_PATH is not configured.")
    model_path = Path(settings.aircraft_detector_onnx_path)
    if not model_path.exists():
        raise RuntimeError(
            f"AIRCRAFT_DETECTOR_ONNX_PATH not found: {model_path}. "
            "Export the YOLO detector to ONNX and update the env path."
        )

    return AircraftDetectionPipeline(
        model_path=model_path,
        confidence_threshold=settings.aircraft_conf_threshold,
        iou_threshold=settings.aircraft_iou_threshold,
        device="auto",
    )


@lru_cache
def get_change_detector() -> ChangeDetectionOnnxPipeline:
    if ChangeDetectionOnnxPipeline is None:
        raise RuntimeError("ChangeDetectionOnnxPipeline unavailable. Check ONNX runtime installation.")
    settings = get_settings()
    if not settings.change_detector_onnx_path:
        raise RuntimeError("CHANGE_DETECTOR_ONNX_PATH is not configured.")
    model_path = Path(settings.change_detector_onnx_path)
    if not model_path.exists():
        raise RuntimeError(
            f"CHANGE_DETECTOR_ONNX_PATH not found: {model_path}. "
            "Export the change detector to ONNX and update the env path."
        )

    return ChangeDetectionOnnxPipeline(
        model_path=model_path,
        device="auto",
    )

