import sys
from pathlib import Path
from datetime import datetime

# Add ml-core to sys.path
repo_root = Path(__file__).resolve().parent.parent
ml_core_dir = repo_root / "ml-core"
if str(ml_core_dir) not in sys.path:
    sys.path.append(str(ml_core_dir))

from intelligence.timeline_engine import TimelineEngine
from intelligence.api_models import Detection

def test_timeline_arrival():
    engine = TimelineEngine(proximity_threshold=20.0)
    t1 = datetime(2026, 3, 7, 12, 0)
    d1 = [Detection(class_name="aircraft", confidence=0.9, bbox=[100, 100, 200, 200], metadata={"lat": 25.20, "lon": 55.27})]
    events = engine.process_detections(d1, t1)
    assert len(events) == 1
    assert events[0].event_type == "AIRCRAFT_ARRIVAL"
