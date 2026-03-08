from __future__ import annotations

import argparse
from pathlib import Path

import torch
import timm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export aircraft classifier checkpoint to ONNX.")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt checkpoint.")
    p.add_argument("--out", type=Path, required=True, help="Output ONNX path.")
    p.add_argument("--model", type=str, default=None, help="timm model name override.")
    p.add_argument("--num-classes", type=int, default=None, help="Number of classes override.")
    p.add_argument("--image-size", type=int, default=None, help="Image size override.")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    ckpt = torch.load(str(args.weights), map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model or cfg.get("model_name")
    if not model_name:
        raise ValueError("Model architecture is required. Pass --model or include config.model_name in checkpoint.")

    num_classes = int(args.num_classes or cfg.get("num_classes") or len(ckpt.get("class_names", [])))
    if num_classes <= 0:
        raise ValueError("num_classes could not be inferred. Pass --num-classes explicitly.")

    image_size = int(args.image_size or cfg.get("image_size") or 224)

    model = timm.create_model(str(model_name), pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"Exported ONNX -> {args.out}")


if __name__ == "__main__":
    main()
