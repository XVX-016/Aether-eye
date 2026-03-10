from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import transform as warp_transform
    from rasterio.io import MemoryFile
except ImportError:  # pragma: no cover - optional dependency
    rasterio = None
    Affine = None
    warp_transform = None
    MemoryFile = None

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover - optional dependency
    CRS = None
    Transformer = None


@dataclass
class GeoContext:
    transform: Any
    crs: str | None
    width: int
    height: int
    bounds: tuple[float, float, float, float] | None = None  # (min_lat, min_lon, max_lat, max_lon)
    tile_id: str | None = None


def _to_affine(transform_value: Any) -> Any:
    if Affine is None:
        return transform_value
    if isinstance(transform_value, Affine):
        return transform_value
    if isinstance(transform_value, (list, tuple)) and len(transform_value) == 6:
        return Affine(*transform_value)
    return transform_value


@lru_cache(maxsize=16)
def _cached_transformer(src_crs: str) -> Any:
    if Transformer is None:
        return None
    if not src_crs or src_crs.upper() in {"EPSG:4326", "WGS84"}:
        return None
    return Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)


def build_transformer(geo_ctx: GeoContext) -> Any:
    if geo_ctx.crs is None:
        return None
    return _cached_transformer(str(geo_ctx.crs))


def pixel_to_latlon(
    x: float,
    y: float,
    geo_ctx: GeoContext,
    transformer: Any | None = None,
) -> tuple[float, float]:
    transform = _to_affine(geo_ctx.transform)
    if rasterio is None or warp_transform is None:
        # Fallback: simple affine math if rasterio unavailable.
        if hasattr(transform, "__mul__"):
            lon, lat = transform * (x, y)
        else:
            a, b, c, d, e, f = transform
            lon = a * x + b * y + c
            lat = d * x + e * y + f
        return float(lat), float(lon)

    lon, lat = rasterio.transform.xy(transform, y, x)
    if transformer is None:
        transformer = build_transformer(geo_ctx)
    if transformer is not None:
        lon, lat = transformer.transform(lon, lat)
    return float(lat), float(lon)


def geo_context_from_bounds(
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
    tile_id: str | None = None,
) -> GeoContext:
    min_lat, min_lon, max_lat, max_lon = bounds
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    res_x = (max_lon - min_lon) / float(width)
    res_y = (min_lat - max_lat) / float(height)  # negative
    transform = (res_x, 0.0, min_lon, 0.0, res_y, max_lat)
    return GeoContext(
        transform=transform,
        crs="EPSG:4326",
        width=int(width),
        height=int(height),
        bounds=bounds,
        tile_id=tile_id,
    )


def read_geotiff_with_context(path: str | Path) -> tuple[np.ndarray, GeoContext]:
    if rasterio is None:
        raise RuntimeError("rasterio is required to read GeoTIFFs.")
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read()
        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3, :, :], (1, 2, 0))
        else:
            rgb = np.transpose(data, (1, 2, 0))
        transform = src.transform
        crs = src.crs.to_string() if src.crs else None
        width = src.width
        height = src.height
        bounds = None
        if src.bounds:
            min_lon, min_lat, max_lon, max_lat = src.bounds
            if crs and crs.upper() not in {"EPSG:4326", "WGS84"} and warp_transform:
                min_lon, min_lat, max_lon, max_lat = rasterio.warp.transform_bounds(
                    crs, "EPSG:4326", min_lon, min_lat, max_lon, max_lat, densify_pts=21
                )
            bounds = (min_lat, min_lon, max_lat, max_lon)
    geo_ctx = GeoContext(
        transform=transform,
        crs=crs,
        width=int(width),
        height=int(height),
        bounds=bounds,
        tile_id=path.stem,
    )
    return rgb, geo_ctx


def read_geotiff_bytes_with_context(data: bytes, tile_id: str | None = None) -> tuple[np.ndarray, GeoContext]:
    if rasterio is None or MemoryFile is None:
        raise RuntimeError("rasterio is required to read GeoTIFFs.")
    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            arr = src.read()
            if arr.shape[0] >= 3:
                rgb = np.transpose(arr[:3, :, :], (1, 2, 0))
            else:
                rgb = np.transpose(arr, (1, 2, 0))
            transform = src.transform
            crs = src.crs.to_string() if src.crs else None
            width = src.width
            height = src.height
            bounds = None
            if src.bounds:
                min_lon, min_lat, max_lon, max_lat = src.bounds
                if crs and crs.upper() not in {"EPSG:4326", "WGS84"} and warp_transform:
                    min_lon, min_lat, max_lon, max_lat = rasterio.warp.transform_bounds(
                        crs, "EPSG:4326", min_lon, min_lat, max_lon, max_lat, densify_pts=21
                    )
                bounds = (min_lat, min_lon, max_lat, max_lon)
    geo_ctx = GeoContext(
        transform=transform,
        crs=crs,
        width=int(width),
        height=int(height),
        bounds=bounds,
        tile_id=tile_id,
    )
    return rgb, geo_ctx
