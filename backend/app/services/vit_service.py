from __future__ import annotations

from functools import lru_cache

from aether_ml import ViTAircraftClassifierPipeline

from app.core.config import get_settings


@lru_cache
def get_vit_aircraft_pipeline() -> ViTAircraftClassifierPipeline:
    settings = get_settings()
    if not settings.vit_aircraft_weights_path:
        raise RuntimeError("VIT_AIRCRAFT_WEIGHTS_PATH is not configured.")

    return ViTAircraftClassifierPipeline(
        weights_path=settings.vit_aircraft_weights_path,
        num_classes=settings.vit_aircraft_num_classes,
        model_name=settings.vit_aircraft_model_name,
        image_size=settings.vit_aircraft_image_size,
        device=None,  # auto
    )

