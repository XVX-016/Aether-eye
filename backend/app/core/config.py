from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BACKEND_PYTHON = str(
    (Path(__file__).resolve().parents[3] / ".venv" / "Scripts" / "python.exe")
)


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
    aircraft_classifier_config_path: str = Field(
        default="backend/configs/inference/aircraft_classifier.yaml",
        alias="AIRCRAFT_CLASSIFIER_CONFIG_PATH",
        description="Path to YAML config describing aircraft classifier checkpoint/architecture.",
    )
    change_detector_config_path: str = Field(
        default="backend/configs/inference/change_detector.yaml",
        alias="CHANGE_DETECTOR_CONFIG_PATH",
        description="Path to YAML config describing change detector ONNX and postprocessing.",
    )
    change_metrics_path: str = Field(
        default="experiments/change/run_01/metrics.json",
        alias="CHANGE_METRICS_PATH",
        description="Path to persisted change training metrics JSON.",
    )

    # STAC ingestion / watcher
    enable_stac_watcher: bool = Field(
        default=False,
        alias="ENABLE_STAC_WATCHER",
        description="Enable background STAC watcher on API startup.",
    )
    stac_poll_minutes: int = Field(
        default=180,
        alias="STAC_POLL_MINUTES",
        description="Polling interval for STAC watcher in minutes.",
    )
    stac_config_path: str = Field(
        default="backend/configs/ingestion/stac.yaml",
        alias="STAC_CONFIG_PATH",
        description="Path to STAC watcher YAML config.",
    )
    stac_aois_json: str = Field(
        default="",
        alias="STAC_AOIS_JSON",
        description="Optional JSON override for AOI list.",
    )
    enable_activity_aggregator: bool = Field(
        default=True,
        alias="ENABLE_ACTIVITY_AGGREGATOR",
        description="Enable aircraft activity aggregation job.",
    )
    enable_intel_fetch_on_startup: bool = Field(
        default=True,
        alias="ENABLE_INTEL_FETCH_ON_STARTUP",
        description="Run the external intel RSS fetch once during API startup.",
    )
    enable_flight_fetch_on_startup: bool = Field(
        default=True,
        alias="ENABLE_FLIGHT_FETCH_ON_STARTUP",
        description="Run the external flight feed fetch once during API startup.",
    )
    activity_window_hours: int = Field(
        default=24,
        alias="ACTIVITY_WINDOW_HOURS",
        description="Window size for activity aggregation (hours).",
    )
    activity_surge_factor: float = Field(
        default=2.0,
        alias="ACTIVITY_SURGE_FACTOR",
        description="Surge factor threshold for activity events.",
    )
    activity_min_count: int = Field(
        default=3,
        alias="ACTIVITY_MIN_COUNT",
        description="Minimum current count required to emit surge.",
    )

    # Dataset roots (Windows defaults)
    satellite_change_root: str = Field(
        default="C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change",
        alias="SATELLITE_CHANGE_ROOT",
        description="Root path for satellite change detection dataset.",
    )
    aircraft_fgvc_root: str = Field(
        default="C:/Computing/Aether-eye/ml_core/DATASET/Aircraft",
        alias="FGVC_AIRCRAFT_ROOT",
        description="Root path for FGVC Aircraft dataset.",
    )

    # Runtime Python enforcement
    backend_python_executable: str = Field(
        default=DEFAULT_BACKEND_PYTHON,
        alias="BACKEND_PYTHON_EXECUTABLE",
        description="Expected Python executable path for backend startup on Windows.",
    )
    enforce_backend_python: bool = Field(
        default=True,
        alias="ENFORCE_BACKEND_PYTHON",
        description="When true on Windows, fail startup if interpreter is not the expected venv Python.",
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

