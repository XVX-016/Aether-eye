import argparse
from pathlib import Path
from tqdm import tqdm
import torch

from aether_ml.config import SiameseChangeConfig
from aether_ml.training.siamese_unet_change import _create_dataloaders, _compute_iou
from aether_ml.models.factory import create_model

def evaluate(config_path: str, model_weights: str, split: str = "val"):
    """
    Evaluates a trained model on the given dataset split.
    """
    print(f"Loading configuration from {config_path}...")
    # In a real scenario, you might load config from YAML. Here we rely on defaults for evaluation.
    config = SiameseChangeConfig(
        root=Path('C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change/LEVIR-CD'),
        train_list=Path('C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change/LEVIR-CD/train.txt'),
        val_list=Path(f'C:/Computing/Aether-eye/ml_core/DATASET/Satellite-Change/LEVIR-CD/{split}.txt'),
        epochs=1,
        batch_size=8,
        num_workers=4,
        model_type='resnet34_unet'
    ).resolved()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    _, val_loader = _create_dataloaders(config)

    # Load model
    model = create_model(config)
    print(f"Loading weights from {model_weights}...")
    checkpoint = torch.load(model_weights, map_location=device)
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    total_iou = 0.0
    total_samples = 0
    total_batches = 0

    print(f"Starting evaluation on {split} split...")
    with torch.no_grad():
        for before, after, mask in tqdm(val_loader, desc=f"eval-{split}"):
            before = before.to(device, non_blocking=True)
            after = after.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Route input through correct model type
            if hasattr(model, 'stem') or 'resnet' in str(type(model)).lower():
                logits = model(before, after)
            else:
                x = torch.cat([before, after], dim=1)
                logits = model(x)
                
            batch_iou = _compute_iou(logits, mask)
            
            bsz = mask.size(0)
            total_iou += batch_iou * bsz
            total_samples += bsz
            total_batches += 1

    mean_iou = total_iou / total_samples if total_samples > 0 else 0.0
    print(f"\nEvaluation Complete!")
    print(f"Total Batches Processed: {total_batches}")
    print(f"Mean IoU: {mean_iou:.4f} (~{mean_iou * 100:.1f}%)")
    
    return mean_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model")
    parser.add_argument("--config", type=str, default="default", help="Path to config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    args = parser.parse_args()

    evaluate(args.config, args.weights, args.split)
