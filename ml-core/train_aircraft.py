from __future__ import annotations

import argparse
from pathlib import Path

from aether_ml.config import FgvcVitConfig
from aether_ml.training.fgvc_vit import train_vit_aircraft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train aircraft classifier with long-tail handling.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to FGVC Aircraft root directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/aircraft/run_02"))
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--imbalance-mode",
        type=str,
        default="weighted_loss",
        choices=["weighted_loss", "sampler", "none"],
        help="How to handle class imbalance.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FgvcVitConfig(
        data_root=args.data_root,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        imbalance_mode=args.imbalance_mode,
        warmup_epochs=args.warmup_epochs,
        mixup_alpha=args.mixup_alpha,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        output_dir=args.output_dir,
        use_amp=not args.no_amp,
    )
    result = train_vit_aircraft(cfg)
    print(result)


if __name__ == "__main__":
    main()
