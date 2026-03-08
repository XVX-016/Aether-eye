from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class DetectionResult(BaseModel):
    detections: List[Detection]
    source_image: str
    processed_at: datetime = datetime.utcnow()

class IntelligenceEvent(BaseModel):
    event_id: str
    event_type: str  # e.g., "AIRCRAFT_ARRIVAL", "AIRCRAFT_DEPARTURE", "CONSTRUCTION_START"
    coordinates: Dict[str, float]  # {"lat": float, "lon": float}
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any] = {}

class TimelineHistory(BaseModel):
    object_id: str
    events: List[IntelligenceEvent]
    first_seen: datetime
    last_seen: datetime
