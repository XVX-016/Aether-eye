from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ChangeRegion:
    region_type: str
    bbox: tuple[int, int, int, int]


def extract_change_regions(mask: np.ndarray, min_area: int = 100) -> list[tuple[int, int, int, int]]:
    """
    Extract connected changed regions from a probability/binary mask.
    Returns bounding boxes as (x1, y1, x2, y2).
    """
    if mask.ndim != 2:
        raise ValueError("Expected 2D mask for region extraction.")

    mask01 = mask.astype(np.float32)
    if mask01.max() > 1.0 or mask01.min() < 0.0:
        mask01 = np.clip(mask01, 0.0, 1.0)

    binary = (mask01 >= 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return []

    num_labels, labels = cv2.connectedComponents(binary)
    boxes: list[tuple[int, int, int, int]] = []

    for label in range(1, num_labels):
        ys, xs = np.where(labels == label)
        if xs.size == 0 or ys.size == 0:
            continue
        area = int(xs.size)
        if area < int(min_area):
            continue
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        boxes.append((x1, y1, x2, y2))

    return boxes
