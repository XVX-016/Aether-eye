from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from fastapi import APIRouter, File, Query, UploadFile
from PIL import Image

from app.api.upload_utils import read_image_bgr
from app.schemas.change_inference import ChangeLatencyResponse, ChangeMetricsResponse
from app.services.change_service import benchmark_change_latency, get_change_metrics


router = APIRouter(prefix="/v1", tags=["change-utils"])

def _encode_png_base64(img_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(img_rgb, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.get(
    "/change/metrics",
    response_model=ChangeMetricsResponse,
    summary="Get persisted training metrics for the active change model.",
)
async def get_change_metrics_v1() -> ChangeMetricsResponse:
    return ChangeMetricsResponse(**get_change_metrics())


@router.get(
    "/change/metrics/latency",
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
    "/change/preview-image",
    summary="Generate a PNG preview for an uploaded image (including GeoTIFF).",
)
async def preview_image(
    image: UploadFile = File(..., description="Input image file (jpeg/png/tiff)."),
) -> dict[str, int | str]:
    bgr = read_image_bgr(image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    return {
        "width": int(w),
        "height": int(h),
        "png_base64": _encode_png_base64(rgb),
    }
