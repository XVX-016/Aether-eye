import sys
from pathlib import Path

# Add ml_core to sys.path
repo_root = Path(__file__).resolve().parent.parent
ml_core_dir = repo_root / "ml_core"
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from intelligence.geo_mapper import pixel_to_geo, calculate_distance

def test_pixel_to_geo():
    transform = [55.2700, 0.00001, 0, 25.2000, 0, -0.00001]
    lat, lon = pixel_to_geo(100, 200, transform)
    assert round(lon, 4) == 55.2710
    assert round(lat, 4) == 25.1980 or round(lat, 4) == 24.1980

def test_calculate_distance():
    dist = calculate_distance(25.20, 55.27, 25.21, 55.28)
    assert dist > 1000
