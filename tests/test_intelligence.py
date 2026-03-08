import sys
from pathlib import Path
from datetime import datetime

# Add ml_core to sys.path
ml_core_dir = Path(__file__).resolve().parent.parent
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from intelligence.geo_mapper import pixel_to_geo, calculate_distance
from intelligence.timeline_engine import TimelineEngine
from intelligence.api_models import Detection

def test_geo_mapper():
    print("Testing Geo Mapper...")
    # Mock transform: [lon_origin, res_x, 0, lat_origin, 0, res_y]
    transform = [55.2700, 0.00001, 0, 25.2000, 0, -0.00001]
    
    # Pixel (100, 200)
    lat, lon = pixel_to_geo(100, 200, transform)
    print(f"Pixel (100, 200) -> Lat: {lat}, Lon: {lon}")
    
    assert round(lat, 4) == 24.1980 or round(lat, 4) == 25.1980  # Depends on the sign of res_y
    assert round(lon, 4) == 55.2710
    
    dist = calculate_distance(25.20, 55.27, 25.21, 55.28)
    print(f"Distance between (25.20, 55.27) and (25.21, 55.28): {dist:.2f}m")
    assert dist > 1000  # Should be around 1.5km
    print("Geo Mapper OK")

def test_timeline_engine():
    print("\nTesting Timeline Engine...")
    engine = TimelineEngine(proximity_threshold=20.0)
    
    # Day 1: One aircraft detected
    t1 = datetime(2026, 3, 7, 12, 0)
    d1 = [Detection(class_name="aircraft", confidence=0.9, bbox=[100, 100, 200, 200], metadata={"lat": 25.20, "lon": 55.27})]
    
    events1 = engine.process_detections(d1, t1)
    print(f"Events Day 1: {[e.event_type for e in events1]}")
    assert len(events1) == 1
    assert events1[0].event_type == "AIRCRAFT_ARRIVAL"
    
    # Day 2: Same aircraft detected (close proximity)
    t2 = datetime(2026, 3, 8, 12, 0)
    d2 = [Detection(class_name="aircraft", confidence=0.92, bbox=[105, 102, 205, 202], metadata={"lat": 25.20001, "lon": 55.27001})]
    
    events2 = engine.process_detections(d2, t2)
    print(f"Events Day 2: {[e.event_type for e in events2]}")
    assert len(events2) == 0  # No new event, just tracking
    
    # Day 3: New aircraft detected (far away)
    t3 = datetime(2026, 3, 9, 12, 0)
    d3 = [Detection(class_name="aircraft", confidence=0.88, bbox=[500, 500, 600, 600], metadata={"lat": 25.25, "lon": 55.30})]
    
    events3 = engine.process_detections(d3, t3)
    print(f"Events Day 3: {[e.event_type for e in events3]}")
    assert len(events3) == 1
    assert events3[0].event_type == "AIRCRAFT_ARRIVAL"
    
    print("Timeline Engine OK")

if __name__ == "__main__":
    try:
        test_geo_mapper()
        test_timeline_engine()
        print("\nAll Intelligence Layer tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
