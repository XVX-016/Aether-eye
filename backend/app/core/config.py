from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Aether Eye API", alias="APP_NAME")
    app_env: str = Field(default="production", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # ONNX model paths
    aircraft_detector_onnx_path: str = Field(
        default="",
        alias="AIRCRAFT_DETECTOR_ONNX_PATH",
        description="Path to exported YOLOv8 ONNX model for aircraft detection.",
    )
    change_detector_onnx_path: str = Field(
        default="",
        alias="CHANGE_DETECTOR_ONNX_PATH",
        description="Path to exported ONNX model for change detection (expects 6-channel input).",
    )

    # Inference thresholds
    aircraft_conf_threshold: float = Field(default=0.25, alias="AIRCRAFT_CONF_THRESHOLD")
    aircraft_iou_threshold: float = Field(default=0.45, alias="AIRCRAFT_IOU_THRESHOLD")

    # ViT aircraft classification + explainability
    vit_aircraft_weights_path: str = Field(
        default="",
        alias="VIT_AIRCRAFT_WEIGHTS_PATH",
        description="Path to fine-tuned ViT weights (.pt) for FGVC Aircraft classification.",
    )
    vit_aircraft_num_classes: int = Field(
        default=100,
        alias="VIT_AIRCRAFT_NUM_CLASSES",
        description="Number of FGVC Aircraft classes (variants).",
    )
    vit_aircraft_model_name: str = Field(
        default="vit_base_patch16_224",
        alias="VIT_AIRCRAFT_MODEL_NAME",
        description="timm model name for ViT architecture.",
    )
    vit_aircraft_image_size: int = Field(
        default=224,
        alias="VIT_AIRCRAFT_IMAGE_SIZE",
        description="Input image size for ViT preprocessing.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

