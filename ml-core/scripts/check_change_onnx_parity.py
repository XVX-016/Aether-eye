from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aether_ml.datasets import LevirChangeDataset
from aether_ml.models.siamese_unet_resnet34 import SiameseUNetResNet34


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Torch vs ONNX parity for change detector.")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args()


def _preprocess(before: Image.Image, after: Image.Image, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    b = tf(before.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    a = tf(after.convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    return b, a


def main() -> None:
    args = parse_args()
    ckpt = torch.load(str(args.weights), map_location="cpu")
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    model = SiameseUNetResNet34(pretrained=False)
    model.load_state_dict(state, strict=False)
    model.eval()

    _ = torch.cuda.is_available()
    sess = ort.InferenceSession(str(args.onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    on = sess.get_outputs()[0].name

    ds = LevirChangeDataset(root=args.data_root, split="val", transform=None)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[: max(1, args.samples)]

    max_abs = 0.0
    mask_match = []
    with torch.no_grad():
        for i in idxs:
            b_img, a_img, _ = ds[i]
            b, a = _preprocess(b_img, a_img, args.image_size)
            tlog = model(torch.from_numpy(b), torch.from_numpy(a)).numpy()
            x = np.concatenate([b, a], axis=1).astype(np.float32)
            olog = sess.run([on], {in_name: x})[0]
            max_abs = max(max_abs, float(np.max(np.abs(tlog - olog))))

            tp = (1.0 / (1.0 + np.exp(-tlog)) > 0.5).astype(np.uint8)
            op = (1.0 / (1.0 + np.exp(-olog)) > 0.5).astype(np.uint8)
            mask_match.append(float((tp == op).mean()))

    print(f"samples={len(idxs)}")
    print(f"max_abs_diff={max_abs:.8f}")
    print(f"mean_mask_match={float(np.mean(mask_match)):.6f}")
    print("providers=" + ",".join(sess.get_providers()))


if __name__ == "__main__":
    main()
