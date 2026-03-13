from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
from rasterio.warp import transform_bounds


DEFAULT_TILE_SIZE = 512


def _resolve_stride(tile_size: int, overlap: int = 0, stride: int | None = None) -> int:
    if stride is not None:
        if stride <= 0:
            raise ValueError("stride must be > 0")
        return stride
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")
    return tile_size - overlap


def _tile_record(dataset: rasterio.DatasetReader, x: int, y: int, tile_size: int) -> dict:
    width = min(tile_size, dataset.width - x)
    height = min(tile_size, dataset.height - y)
    window = Window(x, y, width, height)
    tile = dataset.read(window=window, boundless=True, fill_value=0, out_shape=(dataset.count, tile_size, tile_size))
    transform = window_transform(window, dataset.transform)
    left, bottom, right, top = window_bounds(window, dataset.transform)
    lon_min, lat_min, lon_max, lat_max = transform_bounds(dataset.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)
    return {
        "tile": tile,
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "lat_min": float(lat_min),
        "lat_max": float(lat_max),
        "lon_min": float(lon_min),
        "lon_max": float(lon_max),
        "transform": transform,
    }


def iter_tiles(scene_path: str | Path, tile_size: int = DEFAULT_TILE_SIZE, overlap: int = 0, stride: int | None = None) -> Iterator[dict]:
    scene_path = Path(scene_path)
    stride = _resolve_stride(tile_size, overlap, stride)
    with rasterio.open(scene_path) as dataset:
        for y in range(0, dataset.height, stride):
            for x in range(0, dataset.width, stride):
                yield _tile_record(dataset, x, y, tile_size)


def iter_paired_tiles(
    before_scene_path: str | Path,
    after_scene_path: str | Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = 0,
    stride: int | None = None,
) -> Iterator[dict]:
    before_scene_path = Path(before_scene_path)
    after_scene_path = Path(after_scene_path)
    stride = _resolve_stride(tile_size, overlap, stride)
    with rasterio.open(before_scene_path) as before_ds, rasterio.open(after_scene_path) as after_ds:
        if before_ds.width != after_ds.width or before_ds.height != after_ds.height:
            raise ValueError("before and after scenes must have the same raster dimensions")
        if before_ds.count != after_ds.count:
            raise ValueError("before and after scenes must have the same band count")
        for y in range(0, before_ds.height, stride):
            for x in range(0, before_ds.width, stride):
                before_rec = _tile_record(before_ds, x, y, tile_size)
                after_tile = after_ds.read(
                    window=Window(x, y, before_rec["width"], before_rec["height"]),
                    boundless=True,
                    fill_value=0,
                    out_shape=(after_ds.count, tile_size, tile_size),
                )
                yield {**before_rec, "before_tile": before_rec["tile"], "after_tile": after_tile}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python backend/pipeline/tiler.py <scene.tif>")
        return 1
    path = Path(sys.argv[1])
    tiles = list(iter_tiles(path))
    print(f"tile_count={len(tiles)}")
    if tiles:
        print(f"first_tile_shape={tiles[0]['tile'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
