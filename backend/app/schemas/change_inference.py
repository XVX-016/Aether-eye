from __future__ import annotations

from pydantic import BaseModel, Field


class ChangeInferenceResponse(BaseModel):
    mask_base64: str = Field(..., description="Base64 PNG binary mask (0/255).")
    prob_mask_base64: str = Field(..., description="Base64 PNG probability mask (0-255).")
    change_score: float = Field(..., ge=0.0, le=1.0)
    change_ratio: float = Field(..., ge=0.0, le=1.0)
    changed_pixels: int = Field(..., ge=0)
    overlay_base64: str | None = Field(default=None, description="Optional base64 PNG overlay over after image.")
    debug: dict[str, float] | None = Field(
        default=None, description="Optional debug stats for inputs/logits/masks."
    )
    inference_time_ms: float | None = Field(default=None)
    model_name: str | None = Field(default=None)
    device_used: str | None = Field(default=None)


class ChangeMetricsResponse(BaseModel):
    best_epoch: int = Field(..., ge=1)
    best_val_f1: float = Field(..., ge=0.0, le=1.0)
    best_val_iou: float = Field(..., ge=0.0, le=1.0)
    best_val_precision: float = Field(..., ge=0.0, le=1.0)
    best_val_recall: float = Field(..., ge=0.0, le=1.0)
    best_val_pixel_accuracy: float = Field(..., ge=0.0, le=1.0)


class ChangeLatencyResponse(BaseModel):
    runs: int = Field(..., ge=1)
    mean_ms: float = Field(..., ge=0.0)
    median_ms: float = Field(..., ge=0.0)
    p95_ms: float = Field(..., ge=0.0)
    std_ms: float = Field(..., ge=0.0)
    device_used: str = Field(..., min_length=1)
    input_height: int = Field(..., ge=1)
    input_width: int = Field(..., ge=1)
