from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class AircraftDetectionOut(BaseModel):
    bbox: BBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int = Field(..., ge=0)


class AircraftDetectionsResponse(BaseModel):
    detections: list[AircraftDetectionOut]
    inference_time_ms: float | None = Field(
        default=None, description="End-to-end inference time in milliseconds."
    )
    model_name: str | None = Field(default=None, description="Name or identifier of the model used.")
    device_used: str | None = Field(
        default=None, description="Device label used for inference (e.g. cpu, cuda)."
    )


class ChangeDetectionResponse(BaseModel):
    change_score: float = Field(..., ge=0.0, le=1.0)
    regions: list[dict] | None = Field(
        default=None,
        description="Optional semantic regions with type and bbox when semantic=true.",
    )
    change_mask_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded PNG (grayscale) of the change probability mask.",
    )
    changed_pixels: int | None = Field(
        default=None,
        ge=0,
        description="Optional number of pixels above the configured binary threshold.",
    )
    overlay_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded PNG overlay rendered over the after image.",
    )
    debug: dict[str, float] | None = Field(
        default=None, description="Optional debug stats for inputs/logits/masks."
    )
    inference_time_ms: float | None = Field(
        default=None, description="End-to-end inference time in milliseconds."
    )
    model_name: str | None = Field(default=None, description="Name or identifier of the model used.")
    device_used: str | None = Field(
        default=None, description="Device label used for inference (e.g. cpu, cuda)."
    )

