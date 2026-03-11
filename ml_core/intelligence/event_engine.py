from typing import List, Dict, Optional
from datetime import datetime
from .api_models import Detection, IntelligenceEvent

class EventEngine:
    """
    Combines detections and changes into prioritized high-level intelligence events.
    """
    def __init__(self):
        pass

    def synthesize_events(self, detections: List[Detection], changes: List[Dict]) -> List[IntelligenceEvent]:
        events = []
        
        # Priority Logic:
        # High: Change (new infra) overlaps with a Detection (aircraft).
        # Medium: Detection without historical presence (Arrival).
        # Low: Simple Change detection.
        
        # This is a stub for the complex correlation logic.
        # For now, it converts detections into 'Arrival' events if they are new.
        
        for det in detections:
            # Example: Correlation with 'changes' would happen here.
            priority = "MEDIUM"
            for change in changes:
                # Mock overlap check
                if self._check_overlap(det.bbox, change.get("bbox", [0,0,0,0])):
                    priority = "HIGH"
                    break
            
            events.append(IntelligenceEvent(
                event_id=f"evt_{datetime.now().timestamp()}",
                event_type="AIRCRAFT_ARRIVAL",
                coordinates={"lat": det.metadata.get("lat"), "lon": det.metadata.get("lon")},
                timestamp=datetime.now(),
                confidence=det.confidence,
                metadata={
                    "priority": priority,
                    "object_type": det.class_name
                }
            ))
            
        return events

    def _check_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        # Simple IoU or overlap check stub
        return False
