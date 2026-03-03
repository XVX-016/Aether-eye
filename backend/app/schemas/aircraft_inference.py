from __future__ import annotations

from pydantic import BaseModel, Field


class AircraftInferenceResponse(BaseModel):
    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., description="Predicted aircraft class label.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    origin_country: str = Field(..., description="Country of origin for predicted aircraft class.")
    friend_or_foe: str = Field(..., description="Relationship classification for selected user country.")
    inference_engine: str = Field(..., description="Inference backend used (pytorch or onnxruntime provider).")
    inference_time_ms: float | None = Field(default=None, description="End-to-end inference time in milliseconds.")
    model_name: str | None = Field(default=None, description="Model architecture name.")
    device_used: str | None = Field(default=None, description="Device or provider used for inference.")
