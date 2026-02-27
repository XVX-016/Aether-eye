from __future__ import annotations

import base64
import io
from time import perf_counter

import numpy as np
from PIL import Image

from aether_ml import ChangeDetectionPipeline
from app.schemas.inference import ChangeDetectionRequest, ChangeDetectionResponse


def _decode_base64_image(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    with Image.open(io.BytesIO(raw)) as img:
        return np.array(img.convert("RGB"))


class InferenceService:
    def __init__(self) -> None:
        self.pipeline = ChangeDetectionPipeline()

    def run_change_detection(self, payload: ChangeDetectionRequest) -> ChangeDetectionResponse:
        before = _decode_base64_image(payload.before_image_base64)
        after = _decode_base64_image(payload.after_image_base64)

        start = perf_counter()
        result = self.pipeline.run(before_image=before, after_image=after)
        elapsed_ms = (perf_counter() - start) * 1000.0

        return ChangeDetectionResponse(
            change_score=result.change_score,
            change_mask_base64=None,
            inference_time_ms=elapsed_ms,
            model_name=self.pipeline.model_name,
            device_used=str(self.pipeline.device),
        )


inference_service = InferenceService()

