import sys
import os
from pathlib import Path

# Add ml_core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.training import export_yolov8_onnx


def main() -> None:
  """
  Export a trained YOLOv8 aircraft detector to ONNX.

  Configuration via environment variables:
    YOLO_WEIGHTS_PATH   - path to trained YOLOv8 .pt weights (required)
    YOLO_ONNX_OUTPUT_DIR - directory where ONNX export should be written (optional, default: ./artifacts/onnx)
  """
  repo_root = Path(__file__).resolve().parent.parent.parent.parent

  weights_str = os.environ.get("YOLO_WEIGHTS_PATH")
  if not weights_str:
    raise RuntimeError("YOLO_WEIGHTS_PATH env var must be set to a YOLOv8 .pt weights file.")

  output_dir_str = os.environ.get("YOLO_ONNX_OUTPUT_DIR") or "artifacts/onnx"

  weights_path = (repo_root / weights_str).resolve() if not os.path.isabs(weights_str) else Path(weights_str)
  output_dir = (repo_root / output_dir_str).resolve() if not os.path.isabs(output_dir_str) else Path(output_dir_str)

  onnx_path = export_yolov8_onnx(weights_path=weights_path, output_dir=output_dir)
  print(f"Exported YOLOv8 ONNX model to: {onnx_path}")


if __name__ == "__main__":
  main()

