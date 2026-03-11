from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class AircraftDetection:
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int


class AircraftDetectionPipeline:
    """
    ONNXRuntime-based aircraft detector using an exported YOLOv8 ONNX model.

    Responsibilities:
      - Load exported YOLOv8 ONNX model.
      - Preprocess input image (resize, normalize).
      - Run inference on CPU or CUDA (if available).
      - Apply NMS post-processing.
      - Return structured detections with bbox, confidence, and class_id.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",  # "auto" | "cpu" | "cuda"
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = self.model_path.name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device.lower()

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")

        self.session, self.input_name, self.input_height, self.input_width = self._create_session()

    def _create_session(self) -> tuple[ort.InferenceSession, str, int, int]:
        available_providers = ort.get_available_providers()

        providers: list[str]
        if self.device == "cuda" or self.device == "gpu":
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(
                    "CUDAExecutionProvider requested but not available in onnxruntime installation."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:  # auto
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no inputs.")

        input_tensor = inputs[0]
        shape = input_tensor.shape  # (batch, channels, height, width)
        # Handle dynamic batch dimension (None) gracefully.
        if len(shape) != 4:
            raise RuntimeError(f"Expected 4D input for YOLOv8 ONNX model, got shape: {shape}")

        def _resolve_dim(dim: object, fallback: int) -> int:
            if isinstance(dim, (int, np.integer)):
                return int(dim)
            return fallback

        # For typical YOLOv8 exports, height and width are at indices 2 and 3.
        # Some ONNX exports use dynamic dims like "height"/"width".
        input_height = _resolve_dim(shape[2], 640)
        input_width = _resolve_dim(shape[3], 640)

        return session, input_tensor.name, input_height, input_width

    @property
    def runtime_device(self) -> str:
        """
        Returns a simple device label based on the configured providers.
        """
        providers = self.session.get_providers()
        return "cuda" if "CUDAExecutionProvider" in providers else "cpu"

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Resize and normalize the input image to match the YOLOv8 ONNX model.
        Returns the preprocessed tensor and original H, W.
        """
        if image.ndim == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Assume RGB input; convert to BGR then back to RGB to mirror typical YOLO pipelines.
            img = image[..., ::-1]  # swap channels as needed

        orig_h, orig_w = img.shape[:2]

        img_resized = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype(np.float32) / 255.0

        # Channels first, batch dimension
        img_chw = np.transpose(img_resized, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)

        return img_batch, orig_h, orig_w

    def _nms(
        self,
        boxes: list[list[float]],
        scores: list[float],
        class_ids: list[int],
    ) -> list[AircraftDetection]:
        """
        Apply NMS using OpenCV's dnn.NMSBoxes and return structured detections.
        """
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=self.confidence_threshold,
            nms_threshold=self.iou_threshold,
        )

        detections: list[AircraftDetection] = []
        if len(indices) == 0:
            return detections

        # cv2.dnn.NMSBoxes may return a list of lists or a flat list of indices.
        for idx in indices:
            i = int(idx[0]) if isinstance(idx, (list, tuple, np.ndarray)) else int(idx)
            x, y, w, h = boxes[i]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            detections.append(
                AircraftDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(scores[i]),
                    class_id=int(class_ids[i]),
                )
            )

        return detections

    def detect(self, image: np.ndarray) -> List[AircraftDetection]:
        """
        Run aircraft detection on a single image.

        Args:
            image: Input image as a numpy array (H, W, 3) in RGB or BGR order.

        Returns:
            List of AircraftDetection objects with bbox (x1, y1, x2, y2),
            confidence, and class_id.
        """
        img_input, orig_h, orig_w = self._preprocess(image)

        outputs = self.session.run(
            None,
            {self.input_name: img_input},
        )

        # YOLOv8 ONNX default output is a single array of shape (1, N, 5 + num_classes).
        raw = outputs[0]
        if raw.ndim == 3:
            # (batch, N, C)
            raw = np.squeeze(raw, axis=0)
        elif raw.ndim == 2:
            # Already (N, C)
            pass
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {raw.shape}")

        # Ensure shape (N, C)
        outputs_transposed = np.transpose(raw) if raw.shape[0] < raw.shape[1] else raw
        rows = outputs_transposed.shape[0]

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        x_factor = orig_w / float(self.input_width)
        y_factor = orig_h / float(self.input_height)

        for i in range(rows):
            row = outputs_transposed[i]
            x, y, w, h = row[0], row[1], row[2], row[3]
            class_scores = row[4:]

            if class_scores.size == 0:
                continue

            max_score = float(np.max(class_scores))
            if max_score < self.confidence_threshold:
                continue

            class_id = int(np.argmax(class_scores))

            # YOLOv8 uses center x,y,w,h (in resized image coordinates).
            left = (x - w / 2.0) * x_factor
            top = (y - h / 2.0) * y_factor
            width = w * x_factor
            height = h * y_factor

            boxes.append([left, top, width, height])
            scores.append(max_score)
            class_ids.append(class_id)

        return self._nms(boxes, scores, class_ids)

