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
from app.services.change_service import get_change_detector_config
from app.services.onnx_model_service import get_aircraft_detector, get_change_detector
from ml_inference.geo_projection import read_geotiff_bytes_with_context


router = APIRouter(prefix="/v1", tags=["onnx-inference"])


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


def _normalize_pair_to_uint8(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.dtype == np.uint8 and b.dtype == np.uint8:
        return a, b
    if np.issubdtype(a.dtype, np.unsignedinteger) and np.issubdtype(b.dtype, np.unsignedinteger):
        maxv = float(max(np.iinfo(a.dtype).max, np.iinfo(b.dtype).max))
        if maxv <= 0:
            zeros = np.zeros_like(a, dtype=np.uint8)
            return zeros, np.zeros_like(b, dtype=np.uint8)
        scale = 255.0 / maxv
        a_u8 = np.clip(a.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        b_u8 = np.clip(b.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return a_u8, b_u8

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    if not np.isfinite(a_f).all():
        a_f = np.where(np.isfinite(a_f), a_f, np.nan)
    if not np.isfinite(b_f).all():
        b_f = np.where(np.isfinite(b_f), b_f, np.nan)

    min_a = float(np.nanmin(a_f))
    max_a = float(np.nanmax(a_f))
    min_b = float(np.nanmin(b_f))
    max_b = float(np.nanmax(b_f))
    if 0.0 <= min_a <= 1.0 and 0.0 <= max_a <= 1.0 and 0.0 <= min_b <= 1.0 and 0.0 <= max_b <= 1.0:
        a_u8 = np.clip(a_f * 255.0, 0, 255)
        b_u8 = np.clip(b_f * 255.0, 0, 255)
        return a_u8.astype(np.uint8), b_u8.astype(np.uint8)

    def _robust_bounds(arr: np.ndarray) -> tuple[float, float]:
        try:
            lo, hi = np.nanpercentile(arr, [1, 99])
            return float(lo), float(hi)
        except Exception:
            return float(np.nanmin(arr)), float(np.nanmax(arr))

    lo_a, hi_a = _robust_bounds(a_f)
    lo_b, hi_b = _robust_bounds(b_f)
    minv = min(lo_a, lo_b)
    maxv = max(hi_a, hi_b)
    if not np.isfinite(minv) or not np.isfinite(maxv) or maxv <= minv:
        zeros = np.zeros_like(a_f, dtype=np.uint8)
        return zeros, np.zeros_like(b_f, dtype=np.uint8)

    a_scaled = (a_f - minv) / (maxv - minv)
    b_scaled = (b_f - minv) / (maxv - minv)
    a_scaled = np.nan_to_num(a_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    b_scaled = np.nan_to_num(b_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    a_u8 = np.clip(a_scaled * 255.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip(b_scaled * 255.0, 0, 255).astype(np.uint8)
    return a_u8, b_u8


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


def _read_pair_bgr(before_file: UploadFile, after_file: UploadFile) -> tuple[np.ndarray, np.ndarray]:
    before_data = before_file.file.read()
    after_data = after_file.file.read()
    if not before_data or not after_data:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    before_is_tiff = _is_tiff(before_file)
    after_is_tiff = _is_tiff(after_file)

    if before_is_tiff and after_is_tiff:
        before_rgb, _ = read_geotiff_bytes_with_context(before_data, tile_id=None)
        after_rgb, _ = read_geotiff_bytes_with_context(after_data, tile_id=None)
        before_u8, after_u8 = _normalize_pair_to_uint8(before_rgb, after_rgb)
        return cv2.cvtColor(before_u8, cv2.COLOR_RGB2BGR), cv2.cvtColor(after_u8, cv2.COLOR_RGB2BGR)

    def _decode_other(data: bytes, is_tiff: bool) -> np.ndarray:
        if is_tiff:
            rgb, _ = read_geotiff_bytes_with_context(data, tile_id=None)
            rgb = _normalize_to_uint8(rgb)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        return img

    return _decode_other(before_data, before_is_tiff), _decode_other(after_data, after_is_tiff)


def _mask_to_base64_png(mask: np.ndarray) -> str:
    # mask: float32 [H, W] in [0, 1]
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rgb_to_base64_png(img_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(img_rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _visualize_prob_mask(prob: np.ndarray) -> np.ndarray:
    finite = prob[np.isfinite(prob)]
    if finite.size == 0:
        return np.zeros_like(prob, dtype=np.float32)
    hi = float(np.percentile(finite, 99.5))
    if hi <= 0.0:
        hi = float(finite.max())
    if hi <= 0.0:
        return np.zeros_like(prob, dtype=np.float32)
    vis = np.clip(prob / hi, 0.0, 1.0).astype(np.float32)
    return np.power(vis, 0.35).astype(np.float32)


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
    semantic: bool = Query(default=False, description="Include semantic region extraction and labels."),
    country: str | None = Query(default=None, description="Optional user-selected country."),
    debug: bool = Query(default=False, description="Include debug stats for inputs/logits/masks."),
) -> ChangeDetectionResponse:
    detector = get_change_detector()
    cfg = get_change_detector_config()

    before_bgr, after_bgr = _read_pair_bgr(before_image, after_image)

    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        raise HTTPException(status_code=400, detail="Before and after images must have the same size.")

    start = perf_counter()
    result = detector.run(before_bgr, after_bgr, semantic=semantic, debug=debug)
    elapsed_ms = (perf_counter() - start) * 1000.0

    prob = np.clip(result.change_mask, 0.0, 1.0)
    binary = (prob > cfg.threshold).astype(np.uint8)
    changed_pixels = int(binary.sum())

    mask_b64 = _mask_to_base64_png(prob) if include_mask else None
    overlay_b64 = None
    if include_mask:
        after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)
        red = np.zeros_like(after_rgb, dtype=np.uint8)
        red[..., 0] = 255
        alpha = float(np.clip(cfg.overlay_alpha, 0.0, 1.0))
        mask3 = _visualize_prob_mask(prob)[..., None]
        overlay = (
            after_rgb.astype(np.float32) * (1.0 - alpha * mask3)
            + red.astype(np.float32) * (alpha * mask3)
        ).astype(np.uint8)
        overlay_b64 = _rgb_to_base64_png(overlay)
    regions = (
        [
            {"type": r.region_type, "bbox": [r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]]}
            for r in result.regions
        ]
        if result.regions is not None
        else None
    )
    return ChangeDetectionResponse(
        change_score=result.change_score,
        regions=regions,
        change_mask_base64=mask_b64,
        changed_pixels=changed_pixels,
        overlay_base64=overlay_b64,
        debug=result.debug,
        inference_time_ms=elapsed_ms,
        model_name=detector.model_name,
        device_used=detector.runtime_device,
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

