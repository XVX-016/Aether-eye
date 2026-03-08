from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from aether_ml.models.siamese_unet import SiameseUNetChangeDetector
from aether_ml.pipelines.change_semantic import extract_change_regions, ChangeRegion
from aether_ml.pipelines.region_classifier import RegionClassifier
from contextlib import nullcontext


@dataclass
class ChangeDetectionResult:
    change_score: float
    change_mask: np.ndarray
    regions: list[ChangeRegion] | None = None


class ChangeDetectionPipeline:
    """
    High-level inference pipeline for satellite change detection.
    Handles preprocessing, model inference, and postprocessing.
    """

    def __init__(
        self,
        device: str | None = None,
        weights_path: str | Path | None = None,
        inference_only: bool = True,
        base_channels: int = 32,
    ) -> None:
        """
        Args:
            device: "cpu", CUDA device string, or None to auto-select.
            weights_path: Optional path to a .pt checkpoint for the Siamese U-Net.
            inference_only: If True, the pipeline always runs with gradients disabled
                and the model in eval mode.
            base_channels: Width of the U-Net encoder/decoder.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_only = inference_only

        model = SiameseUNetChangeDetector(in_channels=3, base_channels=base_channels)
        model.to(self.device)

        if weights_path is not None:
            model.load_weights(weights_path, map_location=self.device)

        if self.inference_only:
            model.eval()

        self.model = model
        self.model_name = "siamese-unet-change"

    def _load_and_preprocess(
        self, before: np.ndarray, after: np.ndarray
    ) -> torch.Tensor:
        if before.shape != after.shape:
            raise ValueError("Before and after images must have the same shape.")

        # Ensure 3-channel RGB
        def to_rgb(img: np.ndarray) -> np.ndarray:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img

        before = to_rgb(before)
        after = to_rgb(after)

        # Simple normalization to [0, 1]
        before = before.astype(np.float32) / 255.0
        after = after.astype(np.float32) / 255.0

        # Concatenate along channel dimension: [H, W, 6]
        stacked = np.concatenate([before, after], axis=2).astype(np.float32)
        # Return 6 x H x W tensor on CPU; tiles will be moved to device as needed.
        tensor = torch.from_numpy(stacked).permute(2, 0, 1)  # [6, H, W]
        return tensor

    def run(
        self, before_image: np.ndarray, after_image: np.ndarray, semantic: bool = False, min_area: int = 100
    ) -> ChangeDetectionResult:
        """
        Run change detection on a pair of images.
        """
        stacked = self._load_and_preprocess(before_image, after_image)  # [6, H, W] on CPU
        _, H, W = stacked.shape

        # Sliding window parameters
        tile_size = 512
        overlap = 0.25
        stride = max(1, int(tile_size * (1.0 - overlap)))

        def compute_starts(size: int, tile: int, step: int) -> list[int]:
            if size <= tile:
                return [0]
            starts: list[int] = []
            pos = 0
            while True:
                if pos + tile >= size:
                    starts.append(size - tile)
                    break
                starts.append(pos)
                pos += step
            # ensure uniqueness and sorted order
            return sorted(set(starts))

        y_starts = compute_starts(H, tile_size, stride)
        x_starts = compute_starts(W, tile_size, stride)

        prob_sum = torch.zeros((H, W), dtype=torch.float32)
        count = torch.zeros((H, W), dtype=torch.float32)

        ctx = torch.no_grad() if self.inference_only else nullcontext()

        for y in y_starts:
            y_end = min(y + tile_size, H)
            for x in x_starts:
                x_end = min(x + tile_size, W)

                tile = stacked[:, y:y_end, x:x_end].unsqueeze(0).to(self.device, non_blocking=True)

                with ctx:
                    logits = self.model(tile)  # [1, 1, h, w]

                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu()  # [h, w]
                prob_sum[y:y_end, x:x_end] += probs
                count[y:y_end, x:x_end] += 1.0

        # Avoid division by zero
        count = torch.clamp(count, min=1.0)
        mask = (prob_sum / count).numpy().astype(np.float32)
        change_score = float(mask.mean())
        regions: list[ChangeRegion] | None = None

        if semantic:
            boxes = extract_change_regions(mask, min_area=min_area)
            classifier = RegionClassifier(device=self.device)
            region_items: list[ChangeRegion] = []
            for (x1, y1, x2, y2) in boxes:
                crop = after_image[y1 : y2 + 1, x1 : x2 + 1]
                region_type = classifier.classify_crop(crop)
                region_items.append(ChangeRegion(region_type=region_type, bbox=(x1, y1, x2, y2)))
            regions = region_items

        return ChangeDetectionResult(change_score=change_score, change_mask=mask, regions=regions)

