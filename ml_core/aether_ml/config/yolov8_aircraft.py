from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class YoloV8AircraftConfig:
    """
    Configuration for training YOLOv8 on the xView aircraft subset.
    """

    # Paths
    xview_root: Path
    xview_annotations: Path
    yolo_dataset_dir: Path
    yolo_data_yaml: Path

    # Training options
    model: str = "yolov8n.pt"
    epochs: int = 50
    batch: int = 16
    imgsz: int = 640
    device: str = "0"  # "cpu" or CUDA device string e.g. "0"

    project: Path = Path("runs/train")
    name: str = "yolov8n-xview-aircraft"

    def resolved(self) -> "YoloV8AircraftConfig":
        """
        Convenience helper that resolves all Path fields to absolute paths.
        """
        return YoloV8AircraftConfig(
            xview_root=self.xview_root.expanduser().resolve(),
            xview_annotations=self.xview_annotations.expanduser().resolve(),
            yolo_dataset_dir=self.yolo_dataset_dir.expanduser().resolve(),
            yolo_data_yaml=self.yolo_data_yaml.expanduser().resolve(),
            model=self.model,
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            device=self.device,
            project=self.project.expanduser().resolve(),
            name=self.name,
        )

