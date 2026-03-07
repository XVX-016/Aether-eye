import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Add ml-core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.config import FgvcVitConfig
from aether_ml.training import train_vit_aircraft


def load_config(config_path: Path) -> Dict[str, Any]:
  if config_path.is_file():
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
      raise ValueError(f"Expected mapping at top-level of {config_path}, got {type(data)}")
    return data
  return {}


def main() -> None:
  """
  Entry point for training the ViT aircraft classification model on FGVC Aircraft.

  Dataset paths are configured via:
    - Environment variable:
        FGVC_AIRCRAFT_ROOT
    - or `config.yaml` in the repo root under `aircraft_fgvc.root`
  """
  repo_root = Path(__file__).resolve().parent.parent.parent.parent
  cfg_yaml = load_config(repo_root / "config.yaml")
  ac_cfg = cfg_yaml.get("aircraft_fgvc", {}) or {}

  root_str = os.environ.get("FGVC_AIRCRAFT_ROOT") or ac_cfg.get("root")
  if not root_str:
    raise RuntimeError(
      "Missing configuration for FGVC Aircraft root. "
      "Set FGVC_AIRCRAFT_ROOT env var or aircraft_fgvc.root in config.yaml."
    )

  data_root = (repo_root / root_str).resolve() if not os.path.isabs(root_str) else Path(root_str)

  cfg = FgvcVitConfig(data_root=data_root).resolved()

  metrics = train_vit_aircraft(cfg)
  print("Training finished.")
  for k, v in metrics.items():
    print(f"{k}: {v}")


if __name__ == "__main__":
  main()

