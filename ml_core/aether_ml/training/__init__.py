"""Training entrypoints.

Imports are guarded so individual trainers can run in minimal environments.
"""

try:
    from .yolov8_aircraft import (
        prepare_xview_yolo_dataset,
        train_yolov8_aircraft,
        evaluate_yolov8_aircraft_map,
        export_yolov8_onnx,
    )
except Exception:  # pragma: no cover - runtime environment dependent
    prepare_xview_yolo_dataset = None
    train_yolov8_aircraft = None
    evaluate_yolov8_aircraft_map = None
    export_yolov8_onnx = None

from .fgvc_vit import (
    train_vit_aircraft,
    evaluate_vit_aircraft_top1,
)
from .change_trainer import train_change_unet

try:
    from .siamese_unet_change import (
        train_siamese_unet_change,
        evaluate_siamese_unet_change_iou,
    )
except Exception:  # pragma: no cover - runtime environment dependent
    train_siamese_unet_change = None
    evaluate_siamese_unet_change_iou = None

__all__ = [
    "prepare_xview_yolo_dataset",
    "train_yolov8_aircraft",
    "evaluate_yolov8_aircraft_map",
    "export_yolov8_onnx",
    "train_vit_aircraft",
    "evaluate_vit_aircraft_top1",
    "train_change_unet",
    "train_siamese_unet_change",
    "evaluate_siamese_unet_change_iou",
]
