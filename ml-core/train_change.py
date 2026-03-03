from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from aether_ml.config import ChangeUnetConfig
from aether_ml.training import train_change_unet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese UNet v1 on LEVIR-CD.")
    p.add_argument("--root", type=Path, default=Path("C:/Computing/Aether-eye/ml-core/DATASET/Satellite-Change/LEVIR-CD"))
    p.add_argument("--config", type=Path, default=Path("ml-core/aether_ml/config/change_unet.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/change/run_01"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = {}
    if args.config.is_file():
        cfg_dict = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    cfg = ChangeUnetConfig(
        root=args.root,
        image_size=int(cfg_dict.get("image_size", 256)),
        batch_size=int(cfg_dict.get("batch_size", 8)),
        epochs=int(cfg_dict.get("epochs", 80)),
        learning_rate=float(cfg_dict.get("lr", 1e-4)),
        weight_decay=float(cfg_dict.get("weight_decay", 0.01)),
        early_stopping_patience=int(cfg_dict.get("early_stopping_patience", 10)),
        amp=bool(cfg_dict.get("amp", True)),
        device=args.device,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )
    out = train_change_unet(cfg)
    print(out)


if __name__ == "__main__":
    main()
