import math
from typing import Tuple, Optional, Any

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:
    rasterio = None
    Affine = None

def pixel_to_geo(x: float, y: float, transform: Any) -> Tuple[float, float]:
    """
    Convert pixel (x, y) to geographic (lat, lon) using an affine transform.
    
    The transform is typically [lon_origin, res_x, 0, lat_origin, 0, res_y].
    """
    if hasattr(transform, '__getitem__'):
        # Standard GDAL/Rasterio transform: (c, a, b, f, d, e)
        # lon = c + x*a + y*b
        # lat = f + x*d + y*e
        lon = transform[0] + x * transform[1] + y * transform[2]
        lat = transform[3] + x * transform[4] + y * transform[5]
    else:
        # Assume it's a rasterio.transform.Affine object
        lon, lat = transform * (x, y)
        
    return lat, lon

def get_transform_from_image(image_path: str) -> Any:
    """
    Extract affine transform from a satellite image using Rasterio.
    Returns a default mock transform if extraction fails or rasterio is missing.
    """
    if rasterio:
        try:
            with rasterio.open(image_path) as src:
                return src.transform
        except Exception:
            pass
            
    # Default Mock Transform (Dubai Burj Khalifa area as example)
    # lon_origin=55.27, res_x=0.00001, lat_origin=25.20, res_y=-0.00001
    return [55.2700, 0.00001, 0, 25.2000, 0, -0.00001]

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
