from __future__ import annotations

from pydantic import BaseModel, Field


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
