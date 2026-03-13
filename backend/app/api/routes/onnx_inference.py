from __future__ import annotations

import base64
import io
from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query, HTTPException

from app.schemas.onnx_inference import (
    AircraftDetectionsResponse,
    AircraftDetectionOut,
    BBox,
    ChangeDetectionResponse,
)
from app.api.upload_utils import read_image_bgr, read_pair_bgr
from app.services.change_service import build_change_response
from app.services.onnx_model_service import get_aircraft_detector, get_change_detector


router = APIRouter(prefix="/v1", tags=["onnx-inference"])


@router.post(
    "/aircraft-detection",
    response_model=AircraftDetectionsResponse,
    summary="Run aircraft detection using a YOLOv8 ONNX model.",
)
async def aircraft_detection(
    image: UploadFile = File(..., description="Input image file (jpeg/png/etc)."),
    country: str | None = Query(default=None, description="Optional user-selected country."),
) -> AircraftDetectionsResponse:
    detector = get_aircraft_detector()
    img_bgr = read_image_bgr(image)

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
    semantic: bool = Query(default=False, description="Include semantic region extraction and labels."),
    country: str | None = Query(default=None, description="Optional user-selected country."),
    debug: bool = Query(default=False, description="Include debug stats for inputs/logits/masks."),
) -> ChangeDetectionResponse:
    detector = get_change_detector()

    before_bgr, after_bgr = read_pair_bgr(before_image, after_image)

    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        raise HTTPException(status_code=400, detail="Before and after images must have the same size.")

    start = perf_counter()
    result = build_change_response(
        before_bgr,
        after_bgr,
        include_mask=include_mask,
        semantic=semantic,
        debug=debug,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0

    regions = (
        [
            {"type": r.region_type, "bbox": [r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]]}
            for r in result["regions"]
        ]
        if result["regions"] is not None
        else None
    )
    return ChangeDetectionResponse(
        change_score=result["change_score"],
        regions=regions,
        change_mask_base64=result["change_mask_base64"],
        changed_pixels=result["changed_pixels"],
        overlay_base64=result["overlay_base64"],
        debug=result["debug"],
        inference_time_ms=elapsed_ms,
        model_name=result["model_name"] or detector.model_name,
        device_used=result["device_used"] or detector.runtime_device,
    )


@router.post(
    "/aircraft-detect",
    response_model=AircraftDetectionsResponse,
    summary="Alias endpoint for aircraft detection.",
)
async def aircraft_detect_alias(
    image: UploadFile = File(..., description="Input image file (jpeg/png/etc)."),
    country: str | None = Query(default=None, description="Optional user-selected country."),
) -> AircraftDetectionsResponse:
    return await aircraft_detection(image=image, country=country)

