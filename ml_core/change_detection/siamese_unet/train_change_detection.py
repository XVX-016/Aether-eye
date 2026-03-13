import argparse
import sys
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Add ml_core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.config import SiameseChangeConfig
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
  parser.add_argument("--loss", choices=["hybrid_tversky", "bce_dice"], default=None, help="Override loss function.")
  parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume from checkpoints.")
  parser.add_argument("--no-resize-crop", action="store_true", help="Disable random resized crop augmentation.")
  args = parser.parse_args()

  repo_root = Path(__file__).resolve().parent.parent.parent.parent
  cfg_yaml = load_config(repo_root / "config.yaml")
  sc_cfg = cfg_yaml.get("satellite_change", {}) or {}

  def from_env_or_cfg(env_key: str, *cfg_keys: str) -> str:
    if env_key in os.environ and os.environ[env_key]:
      return os.environ[env_key]
    for k in cfg_keys:
      v = sc_cfg.get(k)
      if v:
        return str(v)
    raise RuntimeError(
      f"Missing configuration for '{env_key}' and keys {cfg_keys} in config.yaml[satellite_change]."
    )

  root_str = from_env_or_cfg("SATELLITE_CHANGE_ROOT", "root")
  train_list_str = from_env_or_cfg("SATELLITE_CHANGE_TRAIN_LIST", "train_list")
  val_list_str = from_env_or_cfg("SATELLITE_CHANGE_VAL_LIST", "val_list")

  root = (repo_root / root_str).resolve() if not os.path.isabs(root_str) else Path(root_str)
  train_list = (root / train_list_str) if not os.path.isabs(train_list_str) else Path(train_list_str)
  val_list = (root / val_list_str) if not os.path.isabs(val_list_str) else Path(val_list_str)

  cfg = SiameseChangeConfig(
    root=root,
    train_list=train_list,
    val_list=val_list,
    epochs=int(args.epochs or os.environ.get("SATELLITE_CHANGE_EPOCHS", sc_cfg.get("epochs", 50))),
    num_workers=int(args.num_workers if args.num_workers is not None else sc_cfg.get("num_workers", 2)),
    output_dir=Path(args.output_dir) if args.output_dir else Path(sc_cfg.get("output_dir", "runs/siamese_unet_change")),
    loss_name=str(args.loss or sc_cfg.get("loss_name", "hybrid_tversky")),
    resume=not args.no_resume,
    use_resize_crop=not args.no_resize_crop,
  ).resolved()

  metrics = train_siamese_unet_change(cfg)
  print("Training finished.")
  for k, v in metrics.items():
    print(f"{k}: {v}")


if __name__ == "__main__":
  main()

