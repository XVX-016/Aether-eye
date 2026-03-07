from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

# Add ml-core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from aether_ml.models.siamese_unet_resnet34 import SiameseUNetResNet34


class ConcatInputWrapper(torch.nn.Module):
    def __init__(self, base: SiameseUNetResNet34) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before = x[:, :3, :, :]
        after = x[:, 3:, :, :]
        return self.base(before, after)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export change detector checkpoint to ONNX.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(str(args.weights), map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt

    base = SiameseUNetResNet34(pretrained=False)
    base.load_state_dict(state, strict=False)
    model = ConcatInputWrapper(base).eval()

    x = torch.randn(1, 6, args.image_size, args.image_size, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        x,
        str(args.out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX -> {args.out}")


if __name__ == "__main__":
    main()
