from __future__ import annotations

import base64
import io
from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from PIL import Image

from app.schemas.onnx_inference import (
    AircraftDetectionsResponse,
    AircraftDetectionOut,
    BBox,
    ChangeDetectionResponse,
)
from app.services.onnx_model_service import get_aircraft_detector, get_change_detector


router = APIRouter(prefix="/v1", tags=["onnx-inference"])


def _read_image_bgr(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _mask_to_base64_png(mask: np.ndarray) -> str:
    # mask: float32 [H, W] in [0, 1]
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post(
    "/aircraft-detection",
    response_model=AircraftDetectionsResponse,
    summary="Run aircraft detection using a YOLOv8 ONNX model.",
)
async def aircraft_detection(
    image: UploadFile = File(..., description="Input image file (jpeg/png/etc)."),
) -> AircraftDetectionsResponse:
    detector = get_aircraft_detector()
    img_bgr = _read_image_bgr(image)

    start = perf_counter()
    detections = detector.detect(img_bgr)
    elapsed_ms = (perf_counter() - start) * 1000.0

    out = [
        AircraftDetectionOut(
            bbox=BBox(x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3]),
            confidence=d.confidence,
            class_id=d.class_id,
        )
        for d in detections
    ]
    return AircraftDetectionsResponse(
        detections=out,
        inference_time_ms=elapsed_ms,
        model_name=detector.model_name,
        device_used=detector.runtime_device,
    )


@router.post(
    "/change-detection",
    response_model=ChangeDetectionResponse,
    summary="Run change detection using an ONNX model on two aligned images.",
)
async def change_detection(
    before_image: UploadFile = File(..., description="Before image file."),
    after_image: UploadFile = File(..., description="After image file."),
    include_mask: bool = Query(default=False, description="Include base64-encoded PNG mask in response."),
) -> ChangeDetectionResponse:
    detector = get_change_detector()

    before_bgr = _read_image_bgr(before_image)
    after_bgr = _read_image_bgr(after_image)

    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        raise HTTPException(status_code=400, detail="Before and after images must have the same size.")

    start = perf_counter()
    result = detector.run(before_bgr, after_bgr)
    elapsed_ms = (perf_counter() - start) * 1000.0

    mask_b64 = _mask_to_base64_png(result.change_mask) if include_mask else None
    return ChangeDetectionResponse(
        change_score=result.change_score,
        change_mask_base64=mask_b64,
        inference_time_ms=elapsed_ms,
        model_name=detector.model_name,
        device_used=detector.runtime_device,
    )

