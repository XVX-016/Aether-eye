from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas.aircraft_inference import AircraftInferenceResponse
from app.services.geopolitics import classify_friend_foe
from app.services.vit_service import (
    classify_aircraft_onnx,
    get_aircraft_classifier_config,
    get_vit_aircraft_pipeline,
)
from ml_inference.geo_projection import read_geotiff_bytes_with_context


router = APIRouter(prefix="/v1", tags=["aircraft-inference"])


def _is_tiff(file: UploadFile) -> bool:
    name = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    return name.endswith((".tif", ".tiff")) or "tiff" in ctype


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.unsignedinteger):
        maxv = float(np.iinfo(img.dtype).max)
        scaled = img.astype(np.float32) / maxv
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    arr = img.astype(np.float32)
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    if 0.0 <= minv and maxv <= 1.0:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if maxv <= minv:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - minv) / (maxv - minv)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _read_image_bgr(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if _is_tiff(file):
        rgb, _geo = read_geotiff_bytes_with_context(data, tile_id=None)
        rgb = _normalize_to_uint8(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


@router.post(
    "/predict/aircraft",
    response_model=AircraftInferenceResponse,
    summary="Production aircraft classifier endpoint (Torch/ONNX).",
)
async def predict_aircraft(
    image: UploadFile = File(..., description="Input image file (jpeg/png/etc)."),
    country: str = Query(default="USA", description="User-selected country for friend/foe relation."),
    use_onnx: bool = Query(default=False, description="Use ONNX runtime instead of PyTorch model."),
) -> AircraftInferenceResponse:
    img_bgr = _read_image_bgr(image)
    cfg = get_aircraft_classifier_config()

    start = perf_counter()
    if use_onnx:
        res, provider = classify_aircraft_onnx(img_bgr)
        device_used = provider
        inference_engine = "onnx"
    else:
        pipeline = get_vit_aircraft_pipeline()
        res = pipeline.classify(img_bgr)
        device_used = pipeline.runtime_device
        inference_engine = "pytorch"
    elapsed_ms = (perf_counter() - start) * 1000.0

    friend_or_foe = classify_friend_foe(country, res.origin_country)
    return AircraftInferenceResponse(
        class_id=res.class_id,
        class_name=res.class_name,
        confidence=res.confidence,
        origin_country=res.origin_country,
        friend_or_foe=friend_or_foe,
        inference_engine=inference_engine,
        inference_time_ms=elapsed_ms,
        model_name=cfg.architecture,
        device_used=device_used,
    )
