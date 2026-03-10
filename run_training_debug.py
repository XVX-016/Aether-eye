import sys
import traceback
from pathlib import Path
from aether_ml.training.siamese_unet_change import train_siamese_unet_change
from aether_ml.config import SiameseChangeConfig

print("Starting training script...")
try:
    config = SiameseChangeConfig(
        root=Path('C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change/LEVIR-CD'),
        train_list=Path('C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change/LEVIR-CD/train.txt'),
        val_list=Path('C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change/LEVIR-CD/val.txt')
    )
    train_siamese_unet_change(config)
except Exception as e:
    print("\n" + "="*50)
    print(f"CRASH DETECTED: {type(e).__name__}")
    print(str(e))
    print("="*50)
    sys.exit(1)
print("Training completed successfully!")
