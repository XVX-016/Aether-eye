import cv2
import numpy as np
from pathlib import Path

import re

class SpectralFilter:
    """
    Stage 4 architecture 'Cheap Mathematical Filter'
    Drops 90-95% of generated satellite tiles before deep learning verification
    by performing ultra-fast pixel-level difference thresholding.
    """
    def __init__(self, diff_threshold: float = 10.0, min_changed_pixels: int = 500):
        # average absolute difference threshold (e.g., 0-255 scale)
        self.diff_threshold = diff_threshold
        self.min_changed_pixels = min_changed_pixels

    def filter_tiles(self, t1_dir: Path, t2_dir: Path):
        """
        Scans matching T1 and T2 tiles in the directories and yields only those
        with significant spectral movement.
        """
        t1_dir = Path(t1_dir)
        t2_dir = Path(t2_dir)
        
        t1_paths = list(t1_dir.glob("*.jpg"))
        print(f"Applying Spectral Filter to {len(t1_paths)} candidate tiles...")
        
        suspicious_tiles = []
        dropped_tiles = 0
        
        for t1_path in t1_paths:
            # Match T2 tile by extracting the coordinate suffix (e.g., _tile_1024_512.jpg)
            match = re.search(r'(_tile_\d+_\d+\.jpg)$', t1_path.name)
            if not match:
                continue
            suffix = match.group(1)
            
            t2_candidates = list(t2_dir.glob(f"*{suffix}"))
            
            if not t2_candidates:
                print(f"Warning: Missing T2 pair for coordinate {suffix}")
                dropped_tiles += 1
                t1_path.unlink(missing_ok=True)
                continue
                
            t2_path = t2_candidates[0]
                
            img_t1 = cv2.imread(str(t1_path))
            img_t2 = cv2.imread(str(t2_path))
            
            if img_t1 is None or img_t2 is None:
                continue

            # 1. Cheap Pixel Math
            # Calculate absolute difference between images
            difference = cv2.absdiff(img_t1, img_t2)
            
            # Simple grayscaling to calculate monolithic difference intensity
            gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            
            # Binary threshold any pixel that changed by more than 'diff_threshold'
            _, thresh = cv2.threshold(gray_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Count the total number of wildly shifted pixels
            changed_pixels = np.count_nonzero(thresh)
            
            if changed_pixels > self.min_changed_pixels:
                suspicious_tiles.append((t1_path, t2_path))
            else:
                dropped_tiles += 1
                # Optional: Delete the dropped tiles from disk to save space
                t1_path.unlink(missing_ok=True)
                t2_path.unlink(missing_ok=True)
                
        print(f"Spectral Filter Complete:")
        print(f"  Suspicious Tiles (Passed): {len(suspicious_tiles)}")
        print(f"  Empty Tiles      (Failed): {dropped_tiles}")
        
        return suspicious_tiles
