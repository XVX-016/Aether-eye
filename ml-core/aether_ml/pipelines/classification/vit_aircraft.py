from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import timm

from aether_ml.explainability.vit_gradcam import ViTGradCam


@dataclass
class ViTClassificationResult:
    class_id: int
    confidence: float


class ViTAircraftClassifierPipeline:
    """
    ViT aircraft classification inference + Grad-CAM explainability.

    Loads a timm Vision Transformer fine-tuned on FGVC Aircraft variants.
    """

    def __init__(
        self,
        weights_path: str | Path,
        num_classes: int,
        model_name: str = "vit_base_patch16_224",
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.num_classes = int(num_classes)
        self.model_name = model_name
        self.image_size = int(image_size)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch.device(self.device)

        self.model = self._build_model()
        self._load_weights(self.weights_path)
        self.model.to(self.torch_device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.1)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        grid_size = self._infer_grid_size()
        self.gradcam_engine = ViTGradCam(model=self.model, grid_size=grid_size)

    def _build_model(self) -> nn.Module:
        return timm.create_model(self.model_name, pretrained=False, num_classes=self.num_classes)

    def _load_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"ViT weights not found at: {path}")
        ckpt = torch.load(str(path), map_location="cpu")
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        self.model.load_state_dict(state_dict, strict=False)

    def _infer_grid_size(self) -> tuple[int, int]:
        # timm ViT usually has patch_embed.grid_size = (H, W) in patches
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "grid_size"):
            gs = patch_embed.grid_size
            return int(gs[0]), int(gs[1])

        # Fallback: derive from image_size and patch_size
        patch_size = 16
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            ps = patch_embed.patch_size
            patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        g = self.image_size // patch_size
        return int(g), int(g)

    def _to_pil_rgb(self, image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.shape[2] == 4:
            return Image.fromarray(image).convert("RGB")
        # Assume BGR (OpenCV default) -> RGB
        rgb = image[..., ::-1]
        return Image.fromarray(rgb.astype(np.uint8)).convert("RGB")

    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        pil = self._to_pil_rgb(image)
        x = self.preprocess(pil).unsqueeze(0)  # [1, 3, H, W]
        return x.to(self.torch_device, non_blocking=True)

    @property
    def runtime_device(self) -> str:
        return self.torch_device.type

    @torch.no_grad()
    def classify(self, image: np.ndarray) -> ViTClassificationResult:
        x = self._prepare_tensor(image)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        conf, cls = torch.max(probs, dim=1)
        return ViTClassificationResult(class_id=int(cls.item()), confidence=float(conf.item()))

    def gradcam(
        self,
        image: np.ndarray,
        target_class: int | None = None,
    ) -> tuple[ViTClassificationResult, np.ndarray]:
        """
        Returns (classification_result, heatmap_float32 [H, W] in [0, 1]).
        """
        x = self._prepare_tensor(image)
        res = self.gradcam_engine.gradcam(x, target_class=target_class)
        cls = ViTClassificationResult(class_id=res.class_id, confidence=res.confidence)
        return cls, res.heatmap

