import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add ml_core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.models.siamese_unet import SiameseUNetV2

class SiameseUNetONNXWrapper(nn.Module):
    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        x = torch.cat([before, after], dim=1)
        return self.base_model(x)

def export_model(weights_path: Path, output_path: Path, image_size: int = 256, opset_version: int = 17) -> None:
    # 1. Instantiate the model (SiameseUNetV2 has base_channels=32 by default)
    model = SiameseUNetV2(in_channels=3, base_channels=32)
    
    # 2. Load weights
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 3. Wrap for separate before/after inputs
    wrapper = SiameseUNetONNXWrapper(model)
    
    # 4. Prepare dummy inputs
    before_dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    after_dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    
    # 5. Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (before_dummy, after_dummy),
        str(output_path),
        input_names=["before", "after"],
        output_names=["change_mask"],
        opset_version=opset_version,
        dynamic_axes={
            "before": {0: "batch"},
            "after": {0: "batch"},
            "change_mask": {0: "batch"}
        }
    )
    print(f"ONNX model successfully exported to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export SiameseUNetV2 model to ONNX with separate before/after inputs.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to PyTorch .pt model weights.")
    parser.add_argument("--out", type=Path, required=True, help="Path to save the output ONNX model.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size for dummy input generation.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    args = parser.parse_args()
    
    export_model(args.weights, args.out, args.image_size, args.opset)

if __name__ == "__main__":
    main()
