from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


class RegionClassifier:
    LABELS = ["Construction", "Vehicle Track", "Terrain Change"]

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _build_model(self) -> torch.nn.Module:
        try:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.LABELS))
        return model

    @torch.no_grad()
    def classify_crop(self, image_crop: np.ndarray) -> str:
        if image_crop is None or image_crop.size == 0:
            return "Terrain Change"

        if image_crop.ndim == 2:
            crop = cv2.cvtColor(image_crop, cv2.COLOR_GRAY2RGB)
        elif image_crop.shape[2] == 4:
            crop = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
        else:
            crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

        x = self.preprocess(crop).unsqueeze(0).to(self.device, non_blocking=True)
        logits = self.model(x)
        idx = int(torch.argmax(logits, dim=1).item())
        if idx < 0 or idx >= len(self.LABELS):
            return "Terrain Change"
        return self.LABELS[idx]
