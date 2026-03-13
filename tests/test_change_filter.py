from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pipeline.change_filter import compute_change_score, compute_ndvi_diff, is_changed


def test_identical_tiles_not_changed():
    tile = np.zeros((4, 8, 8), dtype=np.uint8)
    assert compute_change_score(tile, tile) == 0.0
    assert not is_changed(tile, tile, threshold=0.01)


def test_large_difference_is_changed():
    a = np.zeros((4, 8, 8), dtype=np.uint8)
    b = np.full((4, 8, 8), 255, dtype=np.uint8)
    assert compute_change_score(a, b) > 0.9
    assert is_changed(a, b, threshold=0.05)


def test_threshold_boundary_behavior():
    a = np.zeros((4, 10, 10), dtype=np.float32)
    b = np.full((4, 10, 10), 0.04, dtype=np.float32)
    assert not is_changed(a, b, threshold=0.05)
    assert is_changed(a, b, threshold=0.03)


def test_ndvi_diff_detects_vegetation_change():
    a = np.zeros((4, 4, 4), dtype=np.float32)
    b = np.zeros((4, 4, 4), dtype=np.float32)
    a[2] = 0.2
    a[3] = 0.8
    b[2] = 0.6
    b[3] = 0.2
    assert compute_ndvi_diff(a, b) > 0.1
