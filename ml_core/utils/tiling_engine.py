import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image

class TilingEngine:
    """
    Handles slicing large satellite images into tiles with overlap and stitching predictions.
    """
    def __init__(self, tile_size: int = 256, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def get_tiles(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """
        Slices an image into a list of tiles and their (y, x) top-left coordinates.
        """
        w, h = image.size
        tiles = []
        
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                # Calculate coordinates, ensuring we don't go out of bounds
                # If we're at the edge, we take a tile ending at the edge
                curr_y = min(y, h - self.tile_size) if h > self.tile_size else 0
                curr_x = min(x, w - self.tile_size) if w > self.tile_size else 0
                
                tile = image.crop((curr_x, curr_y, curr_x + self.tile_size, curr_y + self.tile_size))
                tiles.append((tile, (curr_y, curr_x)))
                
                if x + self.tile_size >= w:
                    break
            if y + self.tile_size >= h:
                break
                
        return tiles

    def stitch_masks(self, tiles_with_coords: List[Tuple[np.ndarray, Tuple[int, int]]], original_size: Tuple[int, int]) -> np.ndarray:
        """
        Stitches binary mask tiles back into a single mask, handling overlaps via max (or weighted average).
        tiles_with_coords: List of (mask_tile, (y, x)) where mask_tile is [H, W] or [1, H, W]
        original_size: (H, W)
        """
        h, w = original_size
        full_mask = np.zeros((h, w), dtype=np.float32)
        count_mask = np.zeros((h, w), dtype=np.float32) # For averaging if needed, or just use max
        
        for tile, (y, x) in tiles_with_coords:
            if tile.ndim == 3:
                tile = tile.squeeze()
            
            # Simple max-based stitching for binary masks
            full_mask[y:y+self.tile_size, x:x+self.tile_size] = np.maximum(
                full_mask[y:y+self.tile_size, x:x+self.tile_size], 
                tile
            )
            
        return full_mask

    def process_large_image(self, image: Image.Image, infer_fn) -> np.ndarray:
        """
        End-to-end workflow: tile -> infer -> stitch.
        infer_fn: function that takes a PIL Image and returns a numpy mask.
        """
        tiles = self.get_tiles(image)
        results = []
        
        for tile_img, coords in tiles:
            mask_tile = infer_fn(tile_img)
            results.append((mask_tile, coords))
            
        return self.stitch_masks(results, (image.size[1], image.size[0]))
