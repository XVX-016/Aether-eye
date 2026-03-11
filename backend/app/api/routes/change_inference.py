from __future__ import annotations

from time import perf_counter
import base64
import io

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from app.schemas.change_inference import ChangeInferenceResponse, ChangeLatencyResponse, ChangeMetricsResponse
from app.services.change_service import benchmark_change_latency, get_change_metrics, predict_change
from ml_inference.geo_projection import read_geotiff_bytes_with_context


router = APIRouter(prefix="/v1", tags=["change-inference"])


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


def _encode_png_base64(img_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(img_rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post(
    "/predict/change",
    response_model=ChangeInferenceResponse,
    summary="Predict binary change mask for before/after pair.",
)
async def predict_change_v1(
    before_image: UploadFile = File(..., description="Before image."),
    after_image: UploadFile = File(..., description="After image."),
    include_overlay: bool = Query(default=False, description="Include red overlay image over after image."),
    debug: bool = Query(default=False, description="Include debug stats for inputs/logits/masks."),
) -> ChangeInferenceResponse:
    before_bgr, after_bgr = _read_pair_bgr(before_image, after_image)
    if before_bgr.shape[:2] != after_bgr.shape[:2]:
        raise HTTPException(status_code=400, detail="Before and after images must have same dimensions.")

    t0 = perf_counter()
    out = predict_change(before_bgr, after_bgr, include_overlay=include_overlay, debug=debug)
    elapsed = (perf_counter() - t0) * 1000.0
    return ChangeInferenceResponse(
        mask_base64=out["mask_base64"],
        prob_mask_base64=out["prob_mask_base64"],
        change_score=out["change_score"],
        change_ratio=out["change_ratio"],
        changed_pixels=out["changed_pixels"],
        overlay_base64=out["overlay_base64"],
        debug=out.get("debug"),
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


@router.post(
    "/preview-image",
    summary="Generate a PNG preview for an uploaded image (including GeoTIFF).",
)
async def preview_image(
    image: UploadFile = File(..., description="Input image file (jpeg/png/tiff)."),
) -> dict[str, int | str]:
    data = image.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    if _is_tiff(image):
        rgb, _geo = read_geotiff_bytes_with_context(data, tile_id=None)
        rgb = _normalize_to_uint8(rgb)
    else:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    return {
        "width": int(w),
        "height": int(h),
        "png_base64": _encode_png_base64(rgb),
    }
