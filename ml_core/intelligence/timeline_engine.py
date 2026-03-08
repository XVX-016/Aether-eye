from typing import List, Dict, Optional
from datetime import datetime
from .api_models import Detection, IntelligenceEvent
from .geo_mapper import calculate_distance

class TimelineEngine:
    def __init__(self, proximity_threshold: float = 10.0):
        """
        proximity_threshold: Max distance in meters to consider two detections as the same object.
        """
        self.proximity_threshold = proximity_threshold
        # object_id -> list of detections
        self.history: Dict[str, List[Detection]] = {}

    def process_detections(self, current_detections: List[Detection], timestamp: datetime) -> List[IntelligenceEvent]:
        events = []
        
        # Simple tracking logic: compare current detections with the latest in history
        # In a real system, we'd use Hungarian algorithm or similar.
        
        for det in current_detections:
            matched_id = None
            det_lat, det_lon = det.metadata.get("lat"), det.metadata.get("lon")
            
            if det_lat is not None and det_lon is not None:
                for obj_id, past_dets in self.history.items():
                    last_det = past_dets[-1]
                    last_lat = last_det.metadata.get("lat")
                    last_lon = last_det.metadata.get("lon")
                    
                    if last_lat is not None and last_lon is not None:
                        dist = calculate_distance(det_lat, det_lon, last_lat, last_lon)
                        if dist < self.proximity_threshold:
                            matched_id = obj_id
                            break
            
            if matched_id:
                self.history[matched_id].append(det)
            else:
                # New object detected -> AIRCRAFT_ARRIVAL (if it's an aircraft)
                new_id = f"obj_{len(self.history) + 1}"
                self.history[new_id] = [det]
                
                if det.class_name.lower() == "aircraft":
                    events.append(IntelligenceEvent(
                        event_id=f"evt_{datetime.now().timestamp()}",
                        event_type="AIRCRAFT_ARRIVAL",
                        coordinates={"lat": det_lat, "lon": det_lon},
                        timestamp=timestamp,
                        confidence=det.confidence,
                        metadata={"object_id": new_id, "type": det.class_name}
                    ))
                    
        return events

    def detect_departures(self, current_detections: List[Detection], timestamp: datetime) -> List[IntelligenceEvent]:
        # Logic to identify objects that were present but are now missing
        # For simplicity, we compare the current set of IDs with the active history
        # (This would need a 'keep-alive' or 'visibility' window in a real system)
        return []
