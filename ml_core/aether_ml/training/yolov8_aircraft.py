from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm
from ultralytics import YOLO

from aether_ml.config import YoloV8AircraftConfig
from aether_ml.datasets import XViewAircraftDataset


def prepare_xview_yolo_dataset(cfg: YoloV8AircraftConfig) -> Tuple[Path, Path]:
    """
    Convert the xView aircraft subset into YOLO-format labels and a data YAML.

    Layout created under `cfg.yolo_dataset_dir`:
      images/train
      labels/train
      images/val
      labels/val

    For simplicity this performs a naive 90/10 random split.
    """
    cfg = cfg.resolved()
    root = cfg.xview_root

    ds = XViewAircraftDataset(
        root=root,
        annotations_path=cfg.xview_annotations,
        image_dir=root / "images",
    )

    yolo_root = cfg.yolo_dataset_dir
    images_train = yolo_root / "images" / "train"
    images_val = yolo_root / "images" / "val"
    labels_train = yolo_root / "labels" / "train"
    labels_val = yolo_root / "labels" / "val"

    for p in (images_train, images_val, labels_train, labels_val):
        p.mkdir(parents=True, exist_ok=True)

    import random
    from shutil import copy2

    indices = list(range(len(ds)))
    random.shuffle(indices)
    split_idx = int(0.9 * len(indices))
    train_idx = set(indices[:split_idx])

    # First export all labels; we re-use the helper on dataset.
    ds.to_yolo_txt(labels_train)
    ds.to_yolo_txt(labels_val)

    # Copy corresponding images into train/val folders.
    for i in tqdm(indices, desc="Copying xView images into YOLO layout"):
        sample = ds._samples[i]  # using internal list for efficiency
        dst_dir = images_train if i in train_idx else images_val
        copy2(sample.image_path, dst_dir / sample.image_path.name)

    data_yaml = cfg.yolo_data_yaml
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    num_classes = 1
    yaml_text = "\n".join(
        [
            f"path: {yolo_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            f"nc: {num_classes}",
            "names: [aircraft]",
            "",
        ]
    )
    data_yaml.write_text(yaml_text)

    return yolo_root, data_yaml


def train_yolov8_aircraft(cfg: YoloV8AircraftConfig) -> Dict[str, float]:
    """
    Train YOLOv8 on the prepared xView aircraft dataset.

    Returns a dictionary of metrics from the training run.
    """
    cfg = cfg.resolved()

    model = YOLO(cfg.model)
    results = model.train(
        data=str(cfg.yolo_data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=str(cfg.project),
        name=cfg.name,
    )

    # Ultralytics exposes metrics via the model.metrics attribute after training.
    metrics = getattr(model, "metrics", None)
    metrics_dict: Dict[str, float] = {}

    if metrics is not None:
        # DetMetrics commonly exposes a `results_dict` with keys like:
        # "metrics/mAP50", "metrics/mAP50-95", "metrics/precision(B)", etc.
        results_dict = getattr(metrics, "results_dict", None)
        if results_dict is not None:
            metrics_dict = {str(k): float(v) for k, v in results_dict.items()}

    return metrics_dict


def evaluate_yolov8_aircraft_map(
    weights_path: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
) -> Dict[str, float]:
    """
    Run validation on the aircraft dataset and return mAP metrics.
    """
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
    )

    metrics_dict: Dict[str, float] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if results_dict is not None:
        metrics_dict = {str(k): float(v) for k, v in results_dict.items()}

    return metrics_dict


def export_yolov8_onnx(
    weights_path: str | Path,
    output_dir: str | Path,
    imgsz: int = 640,
    opset: int = 12,
) -> Path:
    """
    Export a trained YOLOv8 model to ONNX.

    Returns the path to the exported ONNX file.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        project=str(output_dir),
        name="onnx",
    )

    return Path(onnx_path)

