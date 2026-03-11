print("BOOTSTRAP: Script started.")
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import sys
import yaml

# Add root and ml_core to path
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "ml_core"))

from aether_ml.config import SiameseChangeConfig
from aether_ml.training.siamese_unet_change import _create_dataloaders, _create_model
from ml_core.evaluation.metrics import compute_metrics

def evaluate(weights_path: str, config_path: str = "config.yaml"):
    print("ENTRY: evaluate() function started.")
    try:
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        change_params = config_dict.get("satellite_change", {})
        # Convert string paths to Path objects
        for key in ["root", "train_list", "val_list", "output_dir"]:
            if key in change_params and isinstance(change_params[key], str):
                change_params[key] = Path(change_params[key])
        
        cfg = SiameseChangeConfig(**change_params)
        cfg = cfg.resolved()
    except Exception as e:
        print(f"Error initializing configuration: {e}")
        import traceback
        traceback.print_exc()
        return
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Load model and data
    _, val_loader = _create_dataloaders(cfg)
    model = _create_model(cfg, device)
    
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    total_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0
    }
    num_samples = 0
    
    with torch.no_grad():
        for before, after, mask in tqdm(val_loader, desc="Evaluating"):
            before = before.to(device)
            after = after.to(device)
            mask = mask.to(device)
            
            x = torch.cat([before, after], dim=1)
            logits = model(x)
            
            metrics = compute_metrics(logits, mask)
            
            bsz = mask.size(0)
            for k in total_metrics:
                total_metrics[k] += metrics[k] * bsz
            num_samples += bsz
            
    # Average
    for k in total_metrics:
        total_metrics[k] /= num_samples
        
    print("\n--- Evaluation Results ---")
    for k, v in total_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print("--------------------------")
    
    return total_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best model weights")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    evaluate(args.weights, args.config)
