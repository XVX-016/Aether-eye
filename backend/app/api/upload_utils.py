from __future__ import annotations

from typing import Tuple, Optional
import json
import numpy as np
import cv2
from fastapi import HTTPException, UploadFile

from ml_inference.geo_projection import read_geotiff_bytes_with_context, GeoContext


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_PIXELS = 4096 * 4096


def http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": code, "message": message})


def is_tiff(file: UploadFile) -> bool:
    name = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    return name.endswith((".tif", ".tiff")) or "tiff" in ctype


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
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


def normalize_pair_to_uint8(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def read_image_bgr(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    if is_tiff(file):
        rgb, _geo = read_geotiff_bytes_with_context(data, tile_id=None)
        rgb = normalize_to_uint8(rgb)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def read_pair_bgr(before_file: UploadFile, after_file: UploadFile) -> tuple[np.ndarray, np.ndarray]:
    before_data = before_file.file.read()
    after_data = after_file.file.read()
    if not before_data or not after_data:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    before_is_tiff = is_tiff(before_file)
    after_is_tiff = is_tiff(after_file)

    if before_is_tiff and after_is_tiff:
        before_rgb, _ = read_geotiff_bytes_with_context(before_data, tile_id=None)
        after_rgb, _ = read_geotiff_bytes_with_context(after_data, tile_id=None)
        before_u8, after_u8 = normalize_pair_to_uint8(before_rgb, after_rgb)
        return cv2.cvtColor(before_u8, cv2.COLOR_RGB2BGR), cv2.cvtColor(after_u8, cv2.COLOR_RGB2BGR)

    def _decode_other(data: bytes, is_tiff_file: bool) -> np.ndarray:
        if is_tiff_file:
            rgb, _ = read_geotiff_bytes_with_context(data, tile_id=None)
            rgb = normalize_to_uint8(rgb)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        return img

    return _decode_other(before_data, before_is_tiff), _decode_other(after_data, after_is_tiff)


def read_upload_image(file: UploadFile) -> tuple[np.ndarray, GeoContext | None, int]:
    data = file.file.read()
    if not data:
        raise http_error(400, "EMPTY_UPLOAD", "Empty file upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise http_error(413, "UPLOAD_TOO_LARGE", "Maximum upload size is 25MB.")
    if is_tiff(file):
        img, geo_ctx = read_geotiff_bytes_with_context(data, tile_id=file.filename)
        h, w = img.shape[:2]
        if h * w > MAX_UPLOAD_PIXELS:
            raise http_error(413, "UPLOAD_TOO_LARGE", "Maximum image resolution is 4096x4096.")
        return img, geo_ctx, len(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise http_error(400, "INVALID_IMAGE", "Invalid image file.")
    h, w = img.shape[:2]
    if h * w > MAX_UPLOAD_PIXELS:
        raise http_error(413, "UPLOAD_TOO_LARGE", "Maximum image resolution is 4096x4096.")
    return img, None, len(data)


def parse_geo_bounds(raw: str | None):
    if not raw:
        return None
    try:
        vals = json.loads(raw) if raw.strip().startswith("[") else [float(x) for x in raw.split(",")]
        if len(vals) != 4:
            return None
        return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
    except Exception:
        return None
