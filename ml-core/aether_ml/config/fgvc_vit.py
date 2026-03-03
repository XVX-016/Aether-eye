from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FgvcVitConfig:
    """
    Configuration for fine-tuning a Vision Transformer on the FGVC Aircraft dataset.
    """

    # Data
    data_root: Path
    model_name: str = "resnet50"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4

    # Optimization
    epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    use_amp: bool = True
    imbalance_mode: str = "weighted_loss"  # "weighted_loss" | "sampler" | "none"
    warmup_epochs: int = 5
    mixup_alpha: float = 0.2
    early_stopping_patience: int = 8

    # Misc
    device: str = "cuda"  # "cpu" or CUDA device string
    output_dir: Path = Path("experiments/aircraft/run_02")
    save_best: bool = True

    def resolved(self) -> "FgvcVitConfig":
        return FgvcVitConfig(
            data_root=self.data_root.expanduser().resolve(),
            model_name=self.model_name,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            label_smoothing=self.label_smoothing,
            use_amp=self.use_amp,
            imbalance_mode=self.imbalance_mode,
            warmup_epochs=self.warmup_epochs,
            mixup_alpha=self.mixup_alpha,
            early_stopping_patience=self.early_stopping_patience,
            device=self.device,
            output_dir=self.output_dir.expanduser().resolve(),
            save_best=self.save_best,
        )

