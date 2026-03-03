"""Public interface for the Aether ML core."""

try:
    from .pipelines.change_detection import ChangeDetectionPipeline, ChangeDetectionResult
except Exception:  # pragma: no cover - runtime environment dependent
    ChangeDetectionPipeline = None
    ChangeDetectionResult = None

try:
    from .pipelines.classification.vit_aircraft import ViTAircraftClassifierPipeline, ViTClassificationResult
except Exception:  # pragma: no cover - runtime environment dependent
    ViTAircraftClassifierPipeline = None
    ViTClassificationResult = None

try:
    from .pipelines.change_semantic import ChangeRegion, extract_change_regions
except Exception:  # pragma: no cover - runtime environment dependent
    ChangeRegion = None
    extract_change_regions = None

try:
    from .pipelines.region_classifier import RegionClassifier
except Exception:  # pragma: no cover - runtime environment dependent
    RegionClassifier = None

try:
    from .pipelines.change_detection_onnx import ChangeDetectionOnnxPipeline
except Exception:  # pragma: no cover - runtime environment dependent
    ChangeDetectionOnnxPipeline = None

try:
    from .pipelines.aircraft_detection import AircraftDetectionPipeline, AircraftDetection
except Exception:  # pragma: no cover - runtime environment dependent
    AircraftDetectionPipeline = None
    AircraftDetection = None

__all__ = [
    "ChangeDetectionPipeline",
    "ChangeDetectionResult",
    "ChangeDetectionOnnxPipeline",
    "AircraftDetectionPipeline",
    "AircraftDetection",
    "ViTAircraftClassifierPipeline",
    "ViTClassificationResult",
    "ChangeRegion",
    "extract_change_regions",
    "RegionClassifier",
]

