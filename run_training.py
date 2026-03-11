import os
from pathlib import Path
from aether_ml.training.siamese_unet_change import train_siamese_unet_change
from aether_ml.config import SiameseChangeConfig

config = SiameseChangeConfig(
    root=Path('C:/Computing/Aether-eye/data/processed/satellite_change/LEVIR-CD'),
    train_list=Path('C:/Computing/Aether-eye/data/processed/satellite_change/LEVIR-CD/train_list.txt'),
    val_list=Path('C:/Computing/Aether-eye/data/processed/satellite_change/LEVIR-CD/val_list.txt'),
    batch_size=4,
    num_workers=2,
    epochs=1
)

print(f"Training with Batch Size: {config.batch_size} | Workers: {config.num_workers}")
print(f"Train exists: {config.train_list.exists()}")

if __name__ == "__main__":
    train_siamese_unet_change(config)
