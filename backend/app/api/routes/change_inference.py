from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.change_inference import ChangeInferenceResponse, ChangeLatencyResponse, ChangeMetricsResponse
from app.services.change_service import benchmark_change_latency, get_change_metrics, predict_change


router = APIRouter(prefix="/v1", tags=["change-inference"])


def _read_image_bgr(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


@router.post(
    "/predict/change",
    response_model=ChangeInferenceResponse,
    summary="Predict binary change mask for before/after pair.",
)
async def predict_change_v1(
    before_image: UploadFile = File(..., description="Before image."),
    after_image: UploadFile = File(..., description="After image."),
    include_overlay: bool = Query(default=False, description="Include red overlay image over after image."),
) -> ChangeInferenceResponse:
    before_bgr = _read_image_bgr(before_image)
    after_bgr = _read_image_bgr(after_image)
    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        raise HTTPException(status_code=400, detail="Before and after images must have same dimensions.")

    t0 = perf_counter()
    out = predict_change(before_bgr, after_bgr, include_overlay=include_overlay)
    elapsed = (perf_counter() - t0) * 1000.0
    return ChangeInferenceResponse(
        mask_base64=out["mask_base64"],
        change_ratio=out["change_ratio"],
        changed_pixels=out["changed_pixels"],
        overlay_base64=out["overlay_base64"],
        inference_time_ms=elapsed,
        model_name=out.get("model_name"),
        device_used=out.get("device_used"),
    )


@router.get(
    "/metrics/change",
    response_model=ChangeMetricsResponse,
    summary="Get persisted training metrics for the active change model.",
)
async def get_change_metrics_v1() -> ChangeMetricsResponse:
    return ChangeMetricsResponse(**get_change_metrics())


@router.get(
    "/metrics/change/latency",
    response_model=ChangeLatencyResponse,
    summary="Benchmark ONNX change model latency over repeated warm runs.",
)
async def get_change_latency_v1(
    runs: int = Query(default=50, ge=1, le=500),
    input_height: int = Query(default=1024, ge=1, le=4096),
    input_width: int = Query(default=1024, ge=1, le=4096),
) -> ChangeLatencyResponse:
    return ChangeLatencyResponse(
        **benchmark_change_latency(runs=runs, input_height=input_height, input_width=input_width)
    )
