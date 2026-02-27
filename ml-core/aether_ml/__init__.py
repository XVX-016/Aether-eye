"""
Public interface for the Aether ML core.
"""

from .pipelines.change_detection import ChangeDetectionPipeline, ChangeDetectionResult
from .pipelines.change_detection_onnx import ChangeDetectionOnnxPipeline
from .pipelines.aircraft_detection import AircraftDetectionPipeline, AircraftDetection
from .pipelines.classification.vit_aircraft import ViTAircraftClassifierPipeline, ViTClassificationResult

__all__ = [
    "ChangeDetectionPipeline",
    "ChangeDetectionResult",
    "ChangeDetectionOnnxPipeline",
    "AircraftDetectionPipeline",
    "AircraftDetection",
    "ViTAircraftClassifierPipeline",
    "ViTClassificationResult",
]

