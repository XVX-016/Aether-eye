from __future__ import annotations

import base64
import io
from time import perf_counter

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from PIL import Image

from app.schemas.vit_explainability import AircraftClassificationResponse, AircraftGradCamResponse
from app.services.geopolitics import classify_friend_foe
from app.services.vit_service import get_aircraft_classifier


router = APIRouter(prefix="/v1", tags=["aircraft-classification"])


def _read_image_bgr(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _heatmap_to_base64_png(heatmap01: np.ndarray, out_w: int, out_h: int) -> str:
    """
    heatmap01: float32 [H, W] in [0,1] (usually model grid size)
    Resizes to out_h/out_w and returns base64 PNG grayscale.
    """
    heat = np.clip(heatmap01, 0.0, 1.0).astype(np.float32)
    heat_resized = cv2.resize(heat, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.clip(heat_resized * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(heat_u8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post(
    "/aircraft-classify",
    response_model=AircraftClassificationResponse,
    summary="Classify aircraft variant (ViT fine-tuned on FGVC Aircraft).",
)
async def aircraft_classify(
    image: UploadFile = File(..., description="Input image file (jpeg/png/etc)."),
    country: str = Query(default="USA", description="User-selected country for geopolitical friend/foe context."),
) -> AircraftClassificationResponse:
    pipeline = get_aircraft_classifier()
    img_bgr = _read_image_bgr(image)

    start = perf_counter()
    res = pipeline.classify(img_bgr)
    elapsed_ms = (perf_counter() - start) * 1000.0

    friend_or_foe = classify_friend_foe(country, res.origin_country)

    return AircraftClassificationResponse(
        class_id=res.class_id,
        class_name=res.class_name,
        confidence=res.confidence,
        origin_country=res.origin_country,
        friend_or_foe=friend_or_foe,
        inference_time_ms=elapsed_ms,
        model_name=pipeline.model_name,
        device_used=pipeline.runtime_device,
    )


@router.post(
    "/aircraft-gradcam",
    response_model=AircraftGradCamResponse,
    summary="Generate Grad-CAM heatmap for ViT aircraft classification.",
)
async def aircraft_gradcam(
    image: UploadFile = File(..., description="Input image file (jpeg/png/etc)."),
    target_class: int | None = Query(
        default=None,
        description="Optional class id to explain. If omitted, explains the top-1 prediction.",
        ge=0,
    ),
) -> AircraftGradCamResponse:
    pipeline = get_aircraft_classifier()
    img_bgr = _read_image_bgr(image)
    h, w = img_bgr.shape[:2]

    start = perf_counter()
    try:
        cls, heatmap = pipeline.gradcam(img_bgr, target_class=target_class)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (perf_counter() - start) * 1000.0

    heatmap_b64 = _heatmap_to_base64_png(heatmap, out_w=w, out_h=h)

    return AircraftGradCamResponse(
        class_id=cls.class_id,
        class_name=cls.class_name,
        confidence=cls.confidence,
        origin_country=cls.origin_country,
        heatmap_base64_png=heatmap_b64,
        inference_time_ms=elapsed_ms,
        model_name=pipeline.model_name,
        device_used=pipeline.runtime_device,
    )

