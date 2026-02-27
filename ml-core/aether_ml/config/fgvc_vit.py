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
    model_name: str = "vit_base_patch16_224"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    # Optimization
    epochs: int = 30
    learning_rate: float = 5e-5
    weight_decay: float = 0.01

    # Misc
    device: str = "cuda"  # "cpu" or CUDA device string
    output_dir: Path = Path("runs/fgvc_vit")
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
            device=self.device,
            output_dir=self.output_dir.expanduser().resolve(),
            save_best=self.save_best,
        )

