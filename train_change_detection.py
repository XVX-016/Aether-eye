from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

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
  repo_root = Path(__file__).resolve().parent
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
  ).resolved()

  metrics = train_siamese_unet_change(cfg)
  print("Training finished.")
  for k, v in metrics.items():
    print(f"{k}: {v}")


if __name__ == "__main__":
  main()

