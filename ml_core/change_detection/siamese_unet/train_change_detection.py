import argparse
import sys
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

# Add ml_core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.config import SiameseChangeConfig, SiameseChangeConfigV3
from aether_ml.training import train_siamese_unet_change


def load_config(config_path: Path) -> Dict[str, Any]:
  if config_path.is_file():
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
      raise ValueError(f"Expected mapping at top-level of {config_path}, got {type(data)}")
    return data
  return {}


def main() -> None:
  """
  Entry point for training the Siamese U-Net change detection model.

  Dataset paths are configured via:
    - Environment variables:
        SATELLITE_CHANGE_ROOT
        SATELLITE_CHANGE_TRAIN_LIST
        SATELLITE_CHANGE_VAL_LIST
    - or `config.yaml` in the repo root under `satellite_change.*`
  """
  parser = argparse.ArgumentParser(description="Train the Siamese U-Net change detector.")
  parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
  parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for checkpoints and metrics.")
  parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count.")
  parser.add_argument(
      "--loss",
      choices=["hybrid_tversky", "bce_dice", "hybrid_tversky_boundary"],
      default=None,
      help="Override loss function."
  )
  parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume from checkpoints.")
  parser.add_argument("--no-resize-crop", action="store_true", help="Disable random resized crop augmentation.")
  parser.add_argument(
      "--config-version",
      type=str,
      default="v2",
      choices=["v2", "v3", "2", "3"],
      help="Config version to use (v2 or v3)."
  )
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parent.parent.parent.parent
  cfg_yaml = load_config(repo_root / "config.yaml")
  sc_cfg = cfg_yaml.get("satellite_change", {}) or {}

  is_v3 = args.config_version in ("v3", "3")
  config_cls = SiameseChangeConfigV3 if is_v3 else SiameseChangeConfig

  # 1. Determine Root Directory
  root_str = os.environ.get("SATELLITE_CHANGE_ROOT")
  if not root_str:
    if is_v3:
      root_str = "data/processed/building_change_v2"
    else:
      root_str = sc_cfg.get("root", "data/processed/building_change")

  # 2. Determine train/val list paths
  train_list_str = os.environ.get("SATELLITE_CHANGE_TRAIN_LIST")
  if not train_list_str:
    if is_v3:
      train_list_str = "train_list.txt"
    else:
      train_list_str = sc_cfg.get("train_list", "train_list.txt")

  val_list_str = os.environ.get("SATELLITE_CHANGE_VAL_LIST")
  if not val_list_str:
    if is_v3:
      val_list_str = "val_list.txt"
    else:
      val_list_str = sc_cfg.get("val_list", "val_list.txt")

  root = (repo_root / root_str).resolve() if not os.path.isabs(root_str) else Path(root_str)
  train_list = (root / train_list_str) if not os.path.isabs(train_list_str) else Path(train_list_str)
  val_list = (root / val_list_str) if not os.path.isabs(val_list_str) else Path(val_list_str)

  # 3. Construct the config object starting from class defaults or explicit overrides
  cfg_kwargs = {
      "root": root,
      "train_list": train_list,
      "val_list": val_list,
  }

  # Override epochs if explicitly provided in args/env/yaml
  epochs_val = args.epochs
  if epochs_val is None:
    if "SATELLITE_CHANGE_EPOCHS" in os.environ:
      epochs_val = int(os.environ["SATELLITE_CHANGE_EPOCHS"])
    elif not is_v3 and "epochs" in sc_cfg:
      epochs_val = int(sc_cfg["epochs"])
  if epochs_val is not None:
    cfg_kwargs["epochs"] = epochs_val

  # Override num_workers
  num_workers_val = args.num_workers
  if num_workers_val is None:
    if not is_v3 and "num_workers" in sc_cfg:
      num_workers_val = int(sc_cfg["num_workers"])
  if num_workers_val is not None:
    cfg_kwargs["num_workers"] = num_workers_val

  # Override output_dir
  output_dir_val = args.output_dir
  if output_dir_val is None:
    if not is_v3 and "output_dir" in sc_cfg:
      output_dir_val = sc_cfg["output_dir"]
  if output_dir_val is not None:
    cfg_kwargs["output_dir"] = Path(output_dir_val)

  # Override loss_name
  loss_name_val = args.loss
  if loss_name_val is None:
    if not is_v3 and "loss_name" in sc_cfg:
      loss_name_val = sc_cfg["loss_name"]
  if loss_name_val is not None:
    cfg_kwargs["loss_name"] = loss_name_val

  # Resume & use_resize_crop behavior
  # For v3, resume defaults to False and use_resize_crop defaults to False.
  cfg_kwargs["resume"] = not args.no_resume
  if args.no_resume:
    cfg_kwargs["resume"] = False

  cfg_kwargs["use_resize_crop"] = not args.no_resize_crop if not is_v3 else False
  if args.no_resize_crop:
    cfg_kwargs["use_resize_crop"] = False

  cfg = config_cls(**cfg_kwargs).resolved()

  metrics = train_siamese_unet_change(cfg)
  print("Training finished.")
  for k, v in metrics.items():
    print(f"{k}: {v}")

  # 4. Copy best v3 checkpoint and export to ONNX
  if is_v3:
    best_model_path = cfg.output_dir / "siamese_unet_change_best.pt"
    if best_model_path.exists():
      dest_dir = repo_root / "ml_core" / "artifacts" / "change_model_v3"
      dest_dir.mkdir(parents=True, exist_ok=True)
      dest_pt = dest_dir / "change_model_v3.pt"
      shutil.copy2(best_model_path, dest_pt)
      print(f"Copied best model to {dest_pt}")

      dest_onnx = dest_dir / "change_model_v3.onnx"
      try:
        sys.path.append(str(repo_root / "ml_core"))
        from scripts.export_change_model import export_model
        export_model(dest_pt, dest_onnx, image_size=cfg.image_size)
        print(f"Auto-packaged ONNX model to {dest_onnx}")
      except Exception as e:
        print(f"Failed to auto-package ONNX model: {e}")


if __name__ == "__main__":
  main()
