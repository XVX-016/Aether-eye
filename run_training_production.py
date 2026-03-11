import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "ml_core"))
from aether_ml.training.siamese_unet_change import train_siamese_unet_change
from aether_ml.config import SiameseChangeConfig

config = SiameseChangeConfig(
    root=Path('C:/Computing/Aether-eye/data/processed/change_detection'),
    train_list=Path('train_list.txt'),
    val_list=Path('val_list.txt'),
    image_size=512,      # Pushed resolution to 512x512 to preserve fine aircraft details (delta-wings) 
    batch_size=4,        # Reduced from 8 to 4 to accommodate the 4x larger memory footprint of 512px
    num_workers=0,       # Disabled multiprocessing completely. Data will load on the main thread safely.
    learning_rate=3e-4,  # Increased starting LR to 3e-4 for optimized ResNet UNet
    epochs=200
)

import torch

if __name__ == "__main__":
    completed_epochs = 0
    checkpoint_path = config.resolved().output_dir / "siamese_unet_change_best.pt"
    if checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            completed_epochs = ckpt.get("epoch", -1) + 1
        except Exception:
            pass

    print("============================================================")
    print(" Starting Siamese U-Net Change Detection Training")
    print(f"   Batch Size:      {config.batch_size}")
    print(f"   Workers:         {config.num_workers}")
    print(f"   Target Epochs:   {config.epochs}")
    if completed_epochs > 0:
        print(f"   Epochs Complete: {completed_epochs} (Resuming from checkpoint)")
    print("============================================================")
    metrics = train_siamese_unet_change(config)
    print("\n============================================================")
    print(" Training Complete!")
    print(f"   Best Validation IoU: {metrics.get('best_val_iou', 0.0):.4f}")
    if "best_model_path" in metrics:
        print(f"   Model Saved To:      {metrics['best_model_path']}")
    print("============================================================")
