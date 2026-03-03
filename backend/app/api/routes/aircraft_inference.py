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


router = APIRouter(prefix="/v1", tags=["aircraft-inference"])


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
