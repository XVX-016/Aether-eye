from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from aether_ml.pipelines.change_semantic import ChangeRegion, extract_change_regions
from aether_ml.pipelines.region_classifier import RegionClassifier


@dataclass
class ChangeDetectionResult:
    change_score: float
    change_mask: np.ndarray
    regions: list[ChangeRegion] | None = None


class ChangeDetectionOnnxPipeline:
    """
    ONNXRuntime-based change detection pipeline.

    Expected model signature:
      input:  (1, 6, H, W) float32, values in [0, 1]
      output: (1, 1, H, W) logits (or probabilities; we handle both defensively)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",  # "auto" | "cpu" | "cuda"
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")

        self.device = device.lower()
        self.session, self.input_name, self.input_height, self.input_width = self._create_session()
        self._region_classifier: RegionClassifier | None = None

    def _create_session(self) -> tuple[ort.InferenceSession, str, int, int]:
        available_providers = ort.get_available_providers()

        providers: list[str]
        if self.device in {"cuda", "gpu"}:
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(
                    "CUDAExecutionProvider requested but not available in onnxruntime installation."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available_providers
                else ["CPUExecutionProvider"]
            )

        session = ort.InferenceSession(str(self.model_path), providers=providers)
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")

        input_tensor = inputs[0]
        shape = input_tensor.shape
        if len(shape) != 4:
            raise RuntimeError(f"Expected 4D input for change detection model, got shape: {shape}")

        input_height = int(shape[2])
        input_width = int(shape[3])
        return session, input_tensor.name, input_height, input_width

    @property
    def runtime_device(self) -> str:
        providers = self.session.get_providers()
        return "cuda" if "CUDAExecutionProvider" in providers else "cpu"

    def _to_rgb(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Assume BGR (OpenCV default) and convert to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _preprocess_pair(self, before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, int, int]:
        if before.shape[:2] != after.shape[:2]:
            raise ValueError("Before and after images must have the same spatial shape.")

        before_rgb = self._to_rgb(before)
        after_rgb = self._to_rgb(after)

        orig_h, orig_w = before_rgb.shape[:2]

        before_resized = cv2.resize(before_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        after_resized = cv2.resize(after_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        before_resized = before_resized.astype(np.float32) / 255.0
        after_resized = after_resized.astype(np.float32) / 255.0

        stacked = np.concatenate([before_resized, after_resized], axis=2)  # H, W, 6
        x = np.transpose(stacked, (2, 0, 1))  # 6, H, W
        x = np.expand_dims(x, axis=0).astype(np.float32)  # 1, 6, H, W

        return x, orig_h, orig_w

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def run(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        semantic: bool = False,
        min_area: int = 100,
    ) -> ChangeDetectionResult:
        x, orig_h, orig_w = self._preprocess_pair(before_image, after_image)
        outputs = self.session.run(None, {self.input_name: x})

        raw = outputs[0]
        if raw.ndim == 4:
            raw = raw[0, 0, :, :]
        elif raw.ndim == 3:
            raw = raw[0, :, :]
        else:
            raise RuntimeError(f"Unexpected ONNX output shape for change mask: {raw.shape}")

        # If the model output already looks like probabilities, keep it; otherwise sigmoid.
        raw_min = float(raw.min())
        raw_max = float(raw.max())
        if raw_min >= 0.0 and raw_max <= 1.0:
            mask_small = raw.astype(np.float32)
        else:
            mask_small = self._sigmoid(raw.astype(np.float32))

        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        change_score = float(mask.mean())
        regions: list[ChangeRegion] | None = None
        if semantic:
            boxes = extract_change_regions(mask, min_area=min_area)
            if self._region_classifier is None:
                self._region_classifier = RegionClassifier(device=self.runtime_device)
            regions = []
            for (x1, y1, x2, y2) in boxes:
                crop = after_image[y1 : y2 + 1, x1 : x2 + 1]
                region_type = self._region_classifier.classify_crop(crop)
                regions.append(ChangeRegion(region_type=region_type, bbox=(x1, y1, x2, y2)))

        return ChangeDetectionResult(change_score=change_score, change_mask=mask, regions=regions)

