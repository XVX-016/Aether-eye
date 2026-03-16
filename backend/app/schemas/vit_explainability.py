from __future__ import annotations

from pydantic import BaseModel, Field


class ViTClassificationResponse(BaseModel):
    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., description="Predicted aircraft class label.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    origin_country: str = Field(..., description="Country of origin for the predicted aircraft class.")
    friend_or_foe: str = Field(..., description="Relationship classification for selected user country.")
    inference_time_ms: float | None = Field(
        default=None, description="End-to-end inference time in milliseconds."
    )
    model_name: str | None = Field(default=None, description="Name or identifier of the model used.")
    device_used: str | None = Field(
        default=None, description="Device label used for inference (e.g. cpu, cuda)."
    )


class ViTGradCamResponse(BaseModel):
    class_id: int = Field(..., ge=0)
    class_name: str = Field(..., description="Predicted aircraft class label.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    origin_country: str = Field(..., description="Country of origin for the predicted aircraft class.")
    heatmap_base64_png: str = Field(
        ...,
        description="Base64-encoded PNG (grayscale) heatmap aligned to the input image size.",
    )
    inference_time_ms: float | None = Field(
        default=None, description="End-to-end inference time in milliseconds."
    )
    model_name: str | None = Field(default=None, description="Name or identifier of the model used.")
    device_used: str | None = Field(
        default=None, description="Device label used for inference (e.g. cpu, cuda)."
    )


AircraftClassificationResponse = ViTClassificationResponse
AircraftGradCamResponse = ViTGradCamResponse

