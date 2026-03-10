import json
import rasterio
from rasterio.windows import Window
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
import numpy as np
import cv2
from pathlib import Path

class TileGenerator:
    """
    Slices massive optical satellite GeoTIFFs into 512x512 ML-ready tensor patches
    using memory-safe windowed reading.
    """
    def __init__(self, tile_size: int = 512):
        self.tile_size = tile_size

    def generate_tiles(self, tif_path: Path, output_dir: Path):
        """
        Reads a GeoTIFF using rasterio windows to avoid RAM exhaustion and writes
        out the non-empty tiles.
        """
        tif_path = Path(tif_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tile_paths = []
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            
            print(f"[{tif_path.name}] Full Resolution: {width}x{height}")
            
            for y in range(0, height, self.tile_size):
                for x in range(0, width, self.tile_size):
                    # Define window and read subset
                    window = Window(x, y, self.tile_size, self.tile_size)
                    
                    # Read the 3 color bands (R,G,B)
                    try:
                        # Sentinel 2 True Color imagery usually has 3 bands (R,G,B) or 4 (RGBA)
                        # We extract the first 3.
                        data = src.read((1, 2, 3), window=window)
                    except Exception as e:
                        print(f"Error reading window {x},{y}: {e}")
                        continue
                    
                    # Convert (C, H, W) to (H, W, C) for OpenCV/PIL
                    img = np.transpose(data, (1, 2, 0))
                    
                    # Check for mostly empty masks/pure black patches
                    if np.mean(img) < 5: 
                        continue
                        
                    # Save the tile
                    # OpenCV expects BGR, so we convert RGB -> BGR
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    tile_id = f"{tif_path.stem}_tile_{x}_{y}"
                    tile_name = f"{tile_id}.jpg"
                    tile_path = output_dir / tile_name
                    cv2.imwrite(str(tile_path), img_bgr)
                    tile_paths.append(tile_path)

                    # Sidecar metadata for geospatial projection
                    tile_transform = rasterio.windows.transform(window, src.transform)
                    tile_h, tile_w = img.shape[:2]
                    tile_bounds = array_bounds(tile_h, tile_w, tile_transform)
                    # array_bounds -> (min_y, min_x, max_y, max_x) in source CRS
                    min_y, min_x, max_y, max_x = tile_bounds
                    crs = src.crs.to_string() if src.crs else None
                    if crs and crs.upper() not in {"EPSG:4326", "WGS84"}:
                        min_x, min_y, max_x, max_y = transform_bounds(
                            crs, "EPSG:4326", min_x, min_y, max_x, max_y, densify_pts=21
                        )

                    sidecar = {
                        "tile_id": tile_id,
                        "crs": crs,
                        "transform": list(tile_transform),
                        "width": int(tile_w),
                        "height": int(tile_h),
                        "tile_bounds": [float(min_y), float(min_x), float(max_y), float(max_x)],
                    }
                    sidecar_path = output_dir / f"{tile_id}.json"
                    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                    
        print(f"[{tif_path.name}] Generated {len(tile_paths)} valid tiles.")
        return tile_paths

if __name__ == "__main__":
    # Ensure this is tested on an actual Geotiff
    pass
