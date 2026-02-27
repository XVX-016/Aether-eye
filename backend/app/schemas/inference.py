from __future__ import annotations

from pydantic import BaseModel, Field


class ChangeDetectionRequest(BaseModel):
    before_image_base64: str = Field(
        ..., description="Base64-encoded image representing the earlier capture."
    )
    after_image_base64: str = Field(
        ..., description="Base64-encoded image representing the later capture."
    )


class ChangeDetectionResponse(BaseModel):
    change_score: float = Field(
        ..., description="Mean change probability across the image, in [0, 1]."
    )
    # In a real system this could be a compressed mask, URL, or tiling reference.
    change_mask_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded change mask for visualization.",
    )
    inference_time_ms: float | None = Field(
        default=None, description="End-to-end inference time in milliseconds."
    )
    model_name: str | None = Field(default=None, description="Name or identifier of the model used.")
    device_used: str | None = Field(
        default=None, description="Device label used for inference (e.g. cpu, cuda)."
    )

