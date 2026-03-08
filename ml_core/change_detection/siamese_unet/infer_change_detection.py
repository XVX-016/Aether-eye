import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

# Add ml_core to sys.path
import sys
ml_core_dir = Path(__file__).resolve().parent.parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.models.siamese_unet import SiameseUNetChangeDetector
from ml_core.utils.tiling_engine import TilingEngine

def infer(model_path, before_path, after_path, output_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SiameseUNetChangeDetector(in_channels=3, base_channels=32)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load images
    before_img = Image.open(before_path).convert("RGB")
    after_img = Image.open(after_path).convert("RGB")
    
    # Normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    def infer_tile(tile_img_before: Image.Image, tile_img_after: Image.Image) -> np.ndarray:
        before_t = F.to_tensor(tile_img_before)
        after_t = F.to_tensor(tile_img_after)
        
        before_t = (before_t - mean) / std
        after_t = (after_t - mean) / std
        
        x = torch.cat([before_t, after_t], dim=0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
            mask = (probs > 0.5).float().cpu().squeeze().numpy()
        return mask

    # Use Tiling Engine
    tiler = TilingEngine(tile_size=256, overlap=64)
    
    # We need to tile both images in sync
    tiles_before = tiler.get_tiles(before_img)
    tiles_after = tiler.get_tiles(after_img)
    
    results = []
    for (tile_b, coords), (tile_a, _) in zip(tiles_before, tiles_after):
        mask_tile = infer_tile(tile_b, tile_a)
        results.append((mask_tile, coords))
        
    full_mask = tiler.stitch_masks(results, (before_img.size[1], before_img.size[0]))
        
    # Save output
    mask_img = Image.fromarray((full_mask * 255).astype(np.uint8))
    mask_img.save(output_path)
    print(f"Inference complete using Tiling Engine. Change map saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--before", type=str, required=True)
    parser.add_argument("--after", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    infer(args.model, args.before, args.after, args.output, args.device)
