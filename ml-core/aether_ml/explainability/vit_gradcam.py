from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


@dataclass
class ViTGradCamResult:
    class_id: int
    confidence: float
    heatmap: np.ndarray  # float32 [H, W] in [0, 1]


class ViTGradCam:
    """
    Grad-CAM for timm Vision Transformer classifiers.

    Uses the recommended target layer `model.blocks[-1].norm1` and a reshape_transform
    that converts patch tokens into a spatial feature map.
    """

    def __init__(self, model: nn.Module, grid_size: Tuple[int, int], target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.grid_h, self.grid_w = grid_size

        self.target_layer = target_layer or getattr(getattr(self.model, "blocks")[-1], "norm1")

        def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
            # tensor: [B, 1 + HW, C] where first token is cls token
            result = tensor[:, 1:, :].reshape(tensor.size(0), self.grid_h, self.grid_w, tensor.size(2))
            # -> [B, C, H, W]
            result = result.permute(0, 3, 1, 2).contiguous()
            return result

        self._cam = GradCAM(model=self.model, target_layers=[self.target_layer], reshape_transform=reshape_transform)

    @torch.no_grad()
    def predict_top1(self, x: torch.Tensor) -> tuple[int, float]:
        """
        Predict top-1 class and confidence from input tensor.
        x: [1, 3, H, W]
        """
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        conf, cls = torch.max(probs, dim=1)
        return int(cls.item()), float(conf.item())

    def gradcam(self, x: torch.Tensor, target_class: int | None = None) -> ViTGradCamResult:
        """
        Compute Grad-CAM heatmap for a given target class (or top-1 if None).
        x: [1, 3, H, W]
        """
        self.model.eval()

        if target_class is None:
            class_id, confidence = self.predict_top1(x)
        else:
            class_id = int(target_class)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                confidence = float(probs[0, class_id].item())

        # grad-cam needs gradients, so do NOT wrap in no_grad
        targets = [ClassifierOutputTarget(class_id)]
        grayscale_cam = self._cam(input_tensor=x, targets=targets)  # [B, H, W]
        heatmap = grayscale_cam[0].astype(np.float32)
        heatmap = np.clip(heatmap, 0.0, 1.0)

        return ViTGradCamResult(class_id=class_id, confidence=confidence, heatmap=heatmap)

