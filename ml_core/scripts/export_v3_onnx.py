import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aether_ml.models.siamese_unet import SiameseUNetV2

# Load checkpoint
ckpt = torch.load(
    "runs/siamese_unet_v3/siamese_unet_change_best.pt",
    map_location="cpu",
    weights_only=False,
)

# Build model (V2 has CBAM attention) and load weights
model = SiameseUNetV2()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Export to ONNX
# Model expects [B, 6, H, W] (before+after concatenated on channel dim)
output_dir = Path("ml_core/artifacts/change_model_v3")
output_dir.mkdir(parents=True, exist_ok=True)

dummy_input = torch.randn(1, 6, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    str(output_dir / "change_model_v3.onnx"),
    input_names=["before_after"],
    output_names=["change_mask"],
    dynamic_axes={
        "before_after": {0: "batch"},
        "change_mask":  {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
onnx_path = output_dir / "change_model_v3.onnx"
print("ONNX exported successfully")
print(f"Size: {onnx_path.stat().st_size / 1e6:.1f} MB")

# Write model card
card = {
    "model": "SiameseUNetV2",
    "version": "v3",
    "checkpoint": "ml_core/artifacts/change_model_v3/change_model_v3.pt",
    "onnx": "ml_core/artifacts/change_model_v3/change_model_v3.onnx",
    "input_format": "[B, 6, H, W] - before(3ch) + after(3ch) concatenated",
    "dataset": "Building-change v2 (expanded)",
    "architecture_notes": "CBAM attention + boundary loss",
    "loss": "hybrid_tversky_boundary",
    "best_val_iou": 0.7366,
    "epoch": ckpt["epoch"],
    "status": "experimental",
    "note": "v2 (0.8243 test IoU) remains production until v3 exceeds it",
}
with open(output_dir / "model_card.json", "w") as f:
    json.dump(card, f, indent=2)
print("Model card written")
