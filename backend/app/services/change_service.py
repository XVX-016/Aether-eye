from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import base64
import io
import json
import time

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from aether_ml import ChangeDetectionOnnxPipeline

from app.core.config import get_settings


@dataclass
class ChangeDetectorConfig:
    onnx_path: str
    metrics_path: str
    threshold: float = 0.5
    overlay_alpha: float = 0.4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    root_candidate = (_repo_root() / p).resolve()
    if root_candidate.exists():
        return root_candidate
    return (Path(__file__).resolve().parents[2] / p).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Change detector config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid change detector YAML structure: {path}")
    return data


@lru_cache
def get_change_detector_config() -> ChangeDetectorConfig:
    settings = get_settings()
    cfg_path = _resolve_path(settings.change_detector_config_path)
    data = _load_yaml(cfg_path)
    onnx_path = str(data.get("onnx_path") or settings.change_detector_onnx_path).strip()
    metrics_path = str(data.get("metrics_path") or settings.change_metrics_path).strip()
    if not onnx_path:
        raise RuntimeError("No change ONNX path configured.")
    if not metrics_path:
        raise RuntimeError("No change metrics path configured.")
    return ChangeDetectorConfig(
        onnx_path=str(_resolve_path(onnx_path)),
        metrics_path=str(_resolve_path(metrics_path)),
        threshold=float(data.get("threshold", 0.5)),
        overlay_alpha=float(data.get("overlay_alpha", 0.4)),
    )


@lru_cache
def get_change_detector_v1() -> ChangeDetectionOnnxPipeline:
    if ChangeDetectionOnnxPipeline is None:
        raise RuntimeError("ChangeDetectionOnnxPipeline unavailable.")
    cfg = get_change_detector_config()
    _ = torch.cuda.is_available()  # preload CUDA DLLs for ORT provider on Windows
    return ChangeDetectionOnnxPipeline(model_path=cfg.onnx_path, device="auto")


def _to_base64_png_gray(mask_u8: np.ndarray) -> str:
    img = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _to_base64_png_rgb(img_rgb: np.ndarray) -> str:
    img = Image.fromarray(img_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_change(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    include_overlay: bool,
    debug: bool = False,
) -> dict[str, Any]:
    cfg = get_change_detector_config()
    model = get_change_detector_v1()
    result = model.run(before_bgr, after_bgr, semantic=False, debug=debug)

    prob = np.clip(result.change_mask, 0.0, 1.0)
    change_score = float(prob.mean())
    prob_mask_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    binary = (prob > cfg.threshold).astype(np.uint8)
    changed_pixels = int(binary.sum())
    total = int(binary.size)
    change_ratio = float(changed_pixels / max(1, total))
    mask_u8 = (binary * 255).astype(np.uint8)
    mask_base64 = _to_base64_png_gray(mask_u8)
    prob_mask_base64 = _to_base64_png_gray(prob_mask_u8)

    overlay_base64 = None
    if include_overlay:
        after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)
        red = np.zeros_like(after_rgb, dtype=np.uint8)
        red[..., 0] = 255
        alpha = float(np.clip(cfg.overlay_alpha, 0.0, 1.0))
        mask3 = prob[..., None].astype(np.float32)
        over = (after_rgb.astype(np.float32) * (1.0 - alpha * mask3) + red.astype(np.float32) * (alpha * mask3)).astype(
            np.uint8
        )
        overlay_base64 = _to_base64_png_rgb(over)

    return {
        "mask_base64": mask_base64,
        "prob_mask_base64": prob_mask_base64,
        "change_score": change_score,
        "change_ratio": change_ratio,
        "changed_pixels": changed_pixels,
        "overlay_base64": overlay_base64,
        "debug": result.debug,
        "model_name": model.model_name,
        "device_used": model.runtime_device,
    }


def get_change_metrics() -> dict[str, float]:
    cfg = get_change_detector_config()
    metrics_path = Path(cfg.metrics_path)
    if not metrics_path.is_file():
        raise RuntimeError(f"Change metrics file not found: {metrics_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    required = [
        "best_epoch",
        "best_val_f1",
        "best_val_iou",
        "best_val_precision",
        "best_val_recall",
        "best_val_pixel_accuracy",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"Change metrics JSON missing keys: {', '.join(missing)}")

    return {
        "best_epoch": int(payload["best_epoch"]),
        "best_val_f1": float(payload["best_val_f1"]),
        "best_val_iou": float(payload["best_val_iou"]),
        "best_val_precision": float(payload["best_val_precision"]),
        "best_val_recall": float(payload["best_val_recall"]),
        "best_val_pixel_accuracy": float(payload["best_val_pixel_accuracy"]),
    }


def benchmark_change_latency(
    runs: int = 50,
    input_height: int = 1024,
    input_width: int = 1024,
) -> dict[str, float | int | str]:
    if runs < 1:
        raise RuntimeError("runs must be >= 1")
    if input_height < 1 or input_width < 1:
        raise RuntimeError("input_height/input_width must be >= 1")

    model = get_change_detector_v1()
    before = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    after = np.zeros((input_height, input_width, 3), dtype=np.uint8)

    # Warmup run to exclude startup overhead from measured distribution.
    _ = model.run(before, after, semantic=False)

    timings_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.run(before, after, semantic=False)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "runs": int(runs),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std(ddof=0)),
        "device_used": str(model.runtime_device),
        "input_height": int(input_height),
        "input_width": int(input_width),
    }
