from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class DetectionRequest(BaseModel):
    image_path: Optional[str] = None
    image_path_before: Optional[str] = None
    image_path_after: Optional[str] = None

class IntelligenceEvent(BaseModel):
    event_id: str
    type: str
    lat: float
    lon: float
    confidence: float
    aircraft_class: Optional[str] = None
    timestamp: datetime
    priority: str
