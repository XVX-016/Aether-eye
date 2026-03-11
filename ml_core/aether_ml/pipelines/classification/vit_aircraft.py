from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
    class_name: str
    confidence: float
    origin_country: str


class ViTAircraftClassifierPipeline:
    """Aircraft classification inference pipeline for timm backbones.

    Backward-compatible class name retained for existing imports/routes.
    Works with ViT/ConvNeXt/ResNet checkpoints produced by train_aircraft.py.
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
        self.model_name = str(model_name)
        self.image_size = int(image_size)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch.device(self.device)

        self.class_names, self.class_to_country = self._load_aircraft_metadata()

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

        # Grad-CAM currently supported only for ViT-like patch models in this project.
        self.gradcam_engine: ViTGradCam | None = None
        if "vit" in self.model_name.lower():
            grid_size = self._infer_grid_size()
            self.gradcam_engine = ViTGradCam(model=self.model, grid_size=grid_size)

    def _build_model(self) -> nn.Module:
        return timm.create_model(self.model_name, pretrained=False, num_classes=self.num_classes)

    def _load_aircraft_metadata(self) -> tuple[list[str], dict[str, str]]:
        metadata_path = Path(__file__).with_name("aircraft_metadata.json")
        if not metadata_path.is_file():
            return [], {}

        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        class_names = list(data.keys())
        class_to_country = {
            name: str(meta.get("country", "Unknown"))
            for name, meta in data.items()
            if isinstance(meta, dict)
        }
        return class_names, class_to_country

    def _load_weights(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Aircraft classifier weights not found at: {path}")

        ckpt = torch.load(str(path), map_location="cpu")
        state_dict = (
            ckpt.get("state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("model")
            or ckpt
        )
        self.model.load_state_dict(state_dict, strict=False)

        # Prefer class names recorded in checkpoint for exact class index mapping.
        ckpt_names = ckpt.get("class_names") if isinstance(ckpt, dict) else None
        if isinstance(ckpt_names, list) and ckpt_names:
            self.class_names = [str(x) for x in ckpt_names]

    def _infer_grid_size(self) -> tuple[int, int]:
        patch_embed = getattr(self.model, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "grid_size"):
            gs = patch_embed.grid_size
            return int(gs[0]), int(gs[1])

        patch_size = 16
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            ps = patch_embed.patch_size
            patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        g = max(1, self.image_size // patch_size)
        return int(g), int(g)

    def _to_pil_rgb(self, image: np.ndarray) -> Image.Image:
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.shape[2] == 4:
            return Image.fromarray(image).convert("RGB")
        rgb = image[..., ::-1]  # BGR -> RGB
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
        class_id = int(cls.item())

        class_name = (
            self.class_names[class_id]
            if 0 <= class_id < len(self.class_names)
            else f"class_{class_id}"
        )
        origin_country = self.class_to_country.get(class_name, "Unknown")
        return ViTClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=float(conf.item()),
            origin_country=origin_country,
        )

    def gradcam(
        self,
        image: np.ndarray,
        target_class: int | None = None,
    ) -> tuple[ViTClassificationResult, np.ndarray]:
        """Returns (classification_result, heatmap float32 [H, W] in [0,1])."""
        if self.gradcam_engine is None:
            raise RuntimeError(
                f"Grad-CAM is only supported for ViT models in this pipeline. Current model: {self.model_name}"
            )

        x = self._prepare_tensor(image)
        base_cls = self.classify(image)
        res = self.gradcam_engine.gradcam(x, target_class=target_class)
        # Use class identity from Grad-CAM target if available.
        class_id = int(res.class_id)
        class_name = (
            self.class_names[class_id]
            if 0 <= class_id < len(self.class_names)
            else base_cls.class_name
        )
        origin_country = self.class_to_country.get(class_name, "Unknown")
        cls = ViTClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=float(res.confidence),
            origin_country=origin_country,
        )
        return cls, res.heatmap
