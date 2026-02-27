from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SiameseChangeConfig:
    """
    Configuration for training a Siamese U-Net on multi-temporal change detection.
    """

    # Data
    root: Path
    train_list: Path
    val_list: Path
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4

    # Model
    base_channels: int = 32

    # Optimization
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Misc
    device: str = "cuda"
    output_dir: Path = Path("runs/siamese_unet_change")
    save_best: bool = True

    def resolved(self) -> "SiameseChangeConfig":
        return SiameseChangeConfig(
            root=self.root.expanduser().resolve(),
            train_list=self.train_list.expanduser().resolve(),
            val_list=self.val_list.expanduser().resolve(),
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            base_channels=self.base_channels,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            output_dir=self.output_dir.expanduser().resolve(),
            save_best=self.save_best,
        )

