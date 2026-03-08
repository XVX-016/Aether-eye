from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChangeUnetConfig:
    # Data
    root: Path
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 4

    # Optimization
    epochs: int = 80
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    early_stopping_patience: int = 10
    amp: bool = True

    # Misc
    device: str = "cuda"
    output_dir: Path = Path("experiments/change/run_01")
    save_best: bool = True

    def resolved(self) -> "ChangeUnetConfig":
        return ChangeUnetConfig(
            root=self.root.expanduser().resolve(),
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            early_stopping_patience=self.early_stopping_patience,
            amp=self.amp,
            device=self.device,
            output_dir=self.output_dir.expanduser().resolve(),
            save_best=self.save_best,
        )
