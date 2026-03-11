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


def read_upload_image(file: UploadFile) -> tuple[np.ndarray, GeoContext | None, int]:
    data = file.file.read()
    if not data:
        raise http_error(400, "EMPTY_UPLOAD", "Empty file upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise http_error(413, "UPLOAD_TOO_LARGE", "Maximum upload size is 25MB.")
    if file.filename and file.filename.lower().endswith((".tif", ".tiff")):
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
