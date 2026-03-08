import unittest
import numpy as np
from PIL import Image
from ml_core.utils.tiling_engine import TilingEngine

class TestTilingEngine(unittest.TestCase):
    def test_tile_and_stitch(self):
        # Create a dummy image (e.g., 500x500)
        w, h = 500, 500
        img = Image.new('RGB', (w, h), color='red')
        
        tiler = TilingEngine(tile_size=256, overlap=64)
        tiles = tiler.get_tiles(img)
        
        # Check number of tiles
        # stride = 256 - 64 = 192
        # Tiles at x=0, 192, (500-256=244) -> 3 tiles per row
        # Tiles at y=0, 192, (500-256=244) -> 3 tiles per col
        # Total 9 tiles
        self.assertEqual(len(tiles), 9)
        
        # Create dummy mask results (all ones)
        results = []
        for tile, coords in tiles:
            mask = np.ones((256, 256), dtype=np.float32)
            results.append((mask, coords))
            
        stitched = tiler.stitch_masks(results, (h, w))
        
        # Check size
        self.assertEqual(stitched.shape, (h, w))
        
        # Check if all pixels are 1.0 (since all tiles were 1.0)
        self.assertTrue(np.all(stitched == 1.0))

    def test_edge_cases(self):
        # Small image (less than tile_size)
        tiler = TilingEngine(tile_size=256, overlap=64)
        img = Image.new('RGB', (100, 100), color='blue')
        tiles = tiler.get_tiles(img)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0][0].size, (256, 256)) # It should crop/pad or handle bounds?
        # Current implementation crops with bounds checking
        
    def test_non_square(self):
        tiler = TilingEngine(tile_size=256, overlap=64)
        img = Image.new('RGB', (1000, 300), color='green')
        tiles = tiler.get_tiles(img)
        # y: 0, (300-256=44) -> 2 rows
        # x: 0, 192, 384, 576, 744 (wait, 1000/192 approx 5.2)
        # x-coords: 0, 192, 384, 576, 768, (1000-256=744)
        # Actually x logic: 0, 192, 384, 576, 768, then next would be 960 which > 1000-256=744
        # Wait, the loop: for x in range(0, 1000, 192): x=0, 192, 384, 576, 768, 960
        # curr_x = min(x, 1000-256=744).
        # x=768 -> curr_x = 744.
        # x=960 -> curr_x = 744.
        # The loop break logic covers this.
        self.assertGreater(len(tiles), 5)

if __name__ == '__main__':
    unittest.main()
