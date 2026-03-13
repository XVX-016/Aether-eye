from __future__ import annotations

import numpy as np


def _to_float(tile: np.ndarray) -> np.ndarray:
    arr = tile.astype(np.float32)
    if arr.size == 0:
        return arr
    maxv = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
    if maxv > 1.0:
        arr = arr / 255.0
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def compute_change_score(tile_t1: np.ndarray, tile_t2: np.ndarray) -> float:
    if tile_t1.shape != tile_t2.shape:
        raise ValueError("tile_t1 and tile_t2 must have the same shape")
    a = _to_float(tile_t1)
    b = _to_float(tile_t2)
    return float(np.mean(np.abs(a - b)))


def compute_ndvi_diff(tile_t1: np.ndarray, tile_t2: np.ndarray) -> float:
    if tile_t1.shape != tile_t2.shape:
        raise ValueError("tile_t1 and tile_t2 must have the same shape")
    if tile_t1.ndim != 3 or tile_t1.shape[0] < 4:
        raise ValueError("NDVI requires at least 4 bands with red and NIR present")

    a = _to_float(tile_t1)
    b = _to_float(tile_t2)
    red_idx = 2
    nir_idx = 3

    ndvi_1 = (a[nir_idx] - a[red_idx]) / (a[nir_idx] + a[red_idx] + 1e-6)
    ndvi_2 = (b[nir_idx] - b[red_idx]) / (b[nir_idx] + b[red_idx] + 1e-6)
    return float(np.mean(np.abs(ndvi_2 - ndvi_1)))


def is_changed(tile_t1: np.ndarray, tile_t2: np.ndarray, threshold: float = 0.05) -> bool:
    return compute_change_score(tile_t1, tile_t2) > float(threshold)
