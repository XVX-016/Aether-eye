from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import timm
import onnxruntime as ort
from torchvision import transforms

# Allow running script directly from repo without installing ml-core wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aether_ml.datasets import FgvcAircraftDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check PyTorch vs ONNX parity for aircraft classifier.")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt checkpoint.")
    p.add_argument("--onnx", type=Path, required=True, help="Path to exported ONNX model.")
    p.add_argument("--data-root", type=Path, required=True, help="FGVC dataset root.")
    p.add_argument("--samples", type=int, default=20, help="Random validation samples for parity check.")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    return p.parse_args()


def _preprocess(image: Image.Image, image_size: int) -> np.ndarray:
    tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    x = tf(image.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    return x


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(str(args.weights), map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_name = args.model or cfg.get("model_name")
    if not model_name:
        raise ValueError("Missing model name. Pass --model or checkpoint config.model_name.")
    num_classes = int(args.num_classes or cfg.get("num_classes") or len(ckpt.get("class_names", [])))
    image_size = int(args.image_size or cfg.get("image_size") or 224)

    model = timm.create_model(str(model_name), pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load torch first so CUDA/cuDNN DLLs are available to ORT CUDA provider on Windows.
    _ = torch.cuda.is_available()
    sess = ort.InferenceSession(
        str(args.onnx),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    ds = FgvcAircraftDataset(root=args.data_root, split="val")
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[: max(1, args.samples)]

    top1_matches = 0
    max_conf_delta = 0.0

    with torch.no_grad():
        for idx in indices:
            pil_img, _ = ds[idx]
            x = _preprocess(pil_img, image_size)
            torch_logits = model(torch.from_numpy(x)).numpy()
            torch_probs = _softmax(torch_logits)
            onnx_logits = sess.run([out_name], {in_name: x})[0]
            onnx_probs = _softmax(np.asarray(onnx_logits, dtype=np.float32))

            torch_top1 = int(np.argmax(torch_probs[0]))
            onnx_top1 = int(np.argmax(onnx_probs[0]))
            if torch_top1 == onnx_top1:
                top1_matches += 1

            torch_conf = float(torch_probs[0, torch_top1])
            onnx_conf = float(onnx_probs[0, onnx_top1])
            max_conf_delta = max(max_conf_delta, abs(torch_conf - onnx_conf))

    n = len(indices)
    match_rate = top1_matches / max(1, n)
    print(f"samples={n}")
    print(f"top1_match_rate={match_rate:.4f}")
    print(f"max_conf_delta={max_conf_delta:.6f}")


if __name__ == "__main__":
    main()
