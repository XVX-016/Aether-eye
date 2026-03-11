from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SiameseChangeConfig:
    """
    Configuration for training a Siamese U-Net on multi-temporal change detection.
    """

    # Data
    root: Path = Path("C:/Computing/Aether-eye/data/processed/change_detection")
    train_list: Path = Path("train_list.txt")
    val_list: Path = Path("val_list.txt")
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 2

    # Model
    model_type: str = "resnet34_unet"
    base_channels: int = 32

    # Optimization
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Misc
    device: str = "cuda"
    output_dir: Path = Path("runs/siamese_unet_change")
    save_best: bool = True

    def resolved(self) -> "SiameseChangeConfig":
        resolved_root = Path(self.root).expanduser().resolve()
        
        # If lists are relative (e.g. "train_list.txt"), resolve them against root
        resolved_train = Path(self.train_list)
        if not resolved_train.is_absolute():
            resolved_train = resolved_root / resolved_train
            
        resolved_val = Path(self.val_list)
        if not resolved_val.is_absolute():
            resolved_val = resolved_root / resolved_val

        return SiameseChangeConfig(
            root=resolved_root,
            train_list=resolved_train,
            val_list=resolved_val,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            model_type=self.model_type,
            base_channels=self.base_channels,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device,
            output_dir=self.output_dir.expanduser().resolve(),
            save_best=self.save_best,
        )

