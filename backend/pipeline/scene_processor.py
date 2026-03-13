from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.database.crud import save_detection, update_scene_status
from app.database.session import async_session
from app.services.change_service import build_change_response
from .change_filter import compute_change_score, is_changed
from .tiler import iter_paired_tiles


logger = logging.getLogger(__name__)


def _tile_to_bgr(tile: np.ndarray) -> np.ndarray:
    if tile.ndim != 3:
        raise ValueError("Expected tile shape (bands, H, W)")
    bands = tile.shape[0]
    if bands == 1:
        rgb = np.repeat(tile[0:1], 3, axis=0)
    elif bands >= 3:
        rgb = tile[:3]
    else:
        pad = np.repeat(tile[-1:], 3 - bands, axis=0)
        rgb = np.concatenate([tile, pad], axis=0)
    hwc = np.moveaxis(rgb, 0, -1)
    if hwc.dtype != np.uint8:
        hwc = hwc.astype(np.float32)
        maxv = float(np.max(hwc)) if hwc.size else 0.0
        if maxv <= 1.0:
            hwc = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
        else:
            hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)


async def process_scene(
    scene_path: str,
    previous_scene_path: str | None,
    *,
    scene_id: str,
    spectral_threshold: float = 0.05,
    overlap: int = 0,
    semantic: bool = False,
    min_change_score: float = 0.05,
) -> list[dict[str, Any]]:
    if not previous_scene_path:
        logger.info("scene_processor: no previous scene for %s; skipping change inference", scene_id)
        async with async_session() as session:
            await update_scene_status(
                session,
                scene_id,
                status="PROCESSED",
                processed=True,
                processed_at=datetime.now(timezone.utc),
            )
            await session.commit()
        return []

    processed_tiles = 0
    skipped_tiles = 0
    detections: list[dict[str, Any]] = []

    async with async_session() as session:
        await update_scene_status(session, scene_id, status="TILED")
        await session.commit()

        for tile in iter_paired_tiles(previous_scene_path, scene_path, tile_size=512, overlap=overlap):
            processed_tiles += 1
            before_tile = tile["before_tile"]
            after_tile = tile["after_tile"]
            spectral_score = compute_change_score(before_tile, after_tile)
            if not is_changed(before_tile, after_tile, threshold=spectral_threshold):
                skipped_tiles += 1
                continue

            before_bgr = _tile_to_bgr(before_tile)
            after_bgr = _tile_to_bgr(after_tile)
            result = build_change_response(
                before_bgr,
                after_bgr,
                include_mask=False,
                semantic=semantic,
                debug=False,
            )
            change_score = float(result["change_score"])
            changed_pixels = int(result["changed_pixels"])
            if change_score <= min_change_score:
                continue

            lat = (float(tile["lat_min"]) + float(tile["lat_max"])) / 2.0
            lon = (float(tile["lon_min"]) + float(tile["lon_max"])) / 2.0
            detection_class = "terrain_change"
            if result["regions"]:
                first = result["regions"][0]
                detection_class = getattr(first, "region_type", detection_class)

            payload = {
                "scene_id": scene_id,
                "tile_x": int(tile["x"]),
                "tile_y": int(tile["y"]),
                "lat": lat,
                "lon": lon,
                "model_type": "change_detection",
                "change_score": change_score,
                "confidence": change_score,
                "detection_class": detection_class,
                "bbox": {
                    "lat_min": float(tile["lat_min"]),
                    "lat_max": float(tile["lat_max"]),
                    "lon_min": float(tile["lon_min"]),
                    "lon_max": float(tile["lon_max"]),
                },
                "metadata_json": {
                    "tile_width": int(tile["width"]),
                    "tile_height": int(tile["height"]),
                    "spectral_score": spectral_score,
                    "changed_pixels": changed_pixels,
                    "source": "scene_processor",
                },
            }
            detections.append(payload)
            await save_detection(session, **payload)

        await update_scene_status(
            session,
            scene_id,
            status="PROCESSED",
            processed=True,
            processed_at=datetime.now(timezone.utc),
        )
        await session.commit()

    logger.info(
        "scene_processor: scene=%s processed_tiles=%s skipped_tiles=%s detections=%s",
        scene_id,
        processed_tiles,
        skipped_tiles,
        len(detections),
    )
    return detections
