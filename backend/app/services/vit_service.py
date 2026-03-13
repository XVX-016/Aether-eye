from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml
try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

from app.core.config import get_settings

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional runtime dependency
    ort = None


@dataclass
class AircraftClassifierConfig:
    model_path: str
    onnx_path: str | None
    architecture: str
    num_classes: int
    image_size: int
    normalization: str = "imagenet"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    # resolve relative to repo root first, then backend root as fallback
    root_candidate = (_repo_root() / p).resolve()
    if root_candidate.exists():
        return root_candidate
    backend_candidate = (Path(__file__).resolve().parents[2] / p).resolve()
    return backend_candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Aircraft classifier config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid aircraft classifier YAML structure: {path}")
    return data


@lru_cache
def get_aircraft_classifier_config() -> AircraftClassifierConfig:
    settings = get_settings()
    cfg_path = _resolve_path(settings.aircraft_classifier_config_path)
    data = _load_yaml(cfg_path)

    # Environment values remain fallback-compatible.
    model_path = str(data.get("model_path") or settings.vit_aircraft_weights_path).strip()
    architecture = str(data.get("architecture") or settings.vit_aircraft_model_name).strip()
    num_classes = int(data.get("num_classes") or settings.vit_aircraft_num_classes)
    image_size = int(data.get("image_size") or settings.vit_aircraft_image_size)
    normalization = str(data.get("normalization") or "imagenet").strip().lower()
    onnx_path_raw = data.get("onnx_path")
    onnx_path = str(onnx_path_raw).strip() if onnx_path_raw else None

    if not model_path:
        raise RuntimeError(
            "No aircraft classifier model path configured. Set model_path in YAML or VIT_AIRCRAFT_WEIGHTS_PATH."
        )

    resolved_model = _resolve_path(model_path)
    resolved_onnx = _resolve_path(onnx_path) if onnx_path else None

    return AircraftClassifierConfig(
        model_path=str(resolved_model),
        onnx_path=str(resolved_onnx) if resolved_onnx is not None else None,
        architecture=architecture,
        num_classes=num_classes,
        image_size=image_size,
        normalization=normalization,
    )


@lru_cache
def get_vit_aircraft_pipeline():
    if torch is None:
        raise RuntimeError("torch is not installed.")
    from aether_ml import ViTAircraftClassifierPipeline

    cfg = get_aircraft_classifier_config()
    return ViTAircraftClassifierPipeline(
        weights_path=cfg.model_path,
        num_classes=cfg.num_classes,
        model_name=cfg.architecture,
        image_size=cfg.image_size,
        device=None,
    )


@lru_cache
def get_aircraft_classifier_onnx_session() -> ort.InferenceSession:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")
    cfg = get_aircraft_classifier_config()
    if not cfg.onnx_path:
        raise RuntimeError("onnx_path is not configured in aircraft classifier YAML.")
    p = Path(cfg.onnx_path)
    if not p.is_file():
        raise RuntimeError(f"Aircraft ONNX model not found: {p}")
    # Ensure CUDA/cuDNN DLLs are loaded from the active torch runtime on Windows.
    # Without this, ORT may silently fall back to CPU due to missing provider DLL deps.
    if torch is not None:
        _ = torch.cuda.is_available()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(p), providers=providers)


def _preprocess_imagenet_rgb(image_bgr: np.ndarray, image_size: int) -> np.ndarray:
    if image_bgr.ndim == 2:
        rgb = np.stack([image_bgr] * 3, axis=-1)
    elif image_bgr.shape[2] == 4:
        rgb = image_bgr[..., :3][:, :, ::-1]
    else:
        rgb = image_bgr[:, :, ::-1]

    pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    resized = pil.resize((int(image_size * 1.1), int(image_size * 1.1)), resample=Image.BILINEAR)
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    crop = resized.crop((left, top, left + image_size, top + image_size))
    arr = np.asarray(crop).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return arr[None, ...].astype(np.float32)  # NCHW


def classify_aircraft_onnx(image_bgr: np.ndarray):
    from aether_ml import ViTClassificationResult

    cfg = get_aircraft_classifier_config()
    session = get_aircraft_classifier_onnx_session()
    torch_pipeline = get_vit_aircraft_pipeline()

    inp = _preprocess_imagenet_rgb(image_bgr, cfg.image_size)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: inp})[0]
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)

    class_id = int(np.argmax(probs[0]))
    conf = float(probs[0, class_id])
    class_name = (
        torch_pipeline.class_names[class_id]
        if 0 <= class_id < len(torch_pipeline.class_names)
        else f"class_{class_id}"
    )
    origin_country = torch_pipeline.class_to_country.get(class_name, "Unknown")

    result = ViTClassificationResult(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        origin_country=origin_country,
    )
    provider = session.get_providers()[0] if session.get_providers() else "onnxruntime"
    return result, provider
