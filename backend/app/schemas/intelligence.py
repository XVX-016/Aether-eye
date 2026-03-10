from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DetectionRequest(BaseModel):
    image_path: Optional[str] = None
    image_path_before: Optional[str] = None
    image_path_after: Optional[str] = None


class IntelligenceProcessRequest(BaseModel):
    image_before_path: Optional[str] = Field(default=None, description="Path to the before image on disk.")
    image_after_path: Optional[str] = Field(default=None, description="Path to the after image on disk.")
    geo_bounds: Optional[List[float]] = Field(
        default=None,
        description="Optional [min_lat, min_lon, max_lat, max_lon] bounds for mapping pixels to lat/lon.",
    )
    run_change_detection: bool = Field(default=True, description="Enable change detection stage.")
    run_aircraft_detection: bool = Field(default=True, description="Enable aircraft detection + classification stage.")
    max_detections: int = Field(default=25, ge=1, le=200, description="Max aircraft detections to process.")
    use_classifier_onnx: bool = Field(default=False, description="Use ONNX classifier for aircraft crops.")

class IntelligenceEvent(BaseModel):
    event_id: str
    type: str
    lat: float
    lon: float
    confidence: float
    priority: str
    timestamp: datetime
    bbox: Optional[List[float]] = None
    source: Optional[str] = None
    tile_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    geometry: Optional[Dict[str, Any]] = None


class IntelligenceProcessResponse(BaseModel):
    events: List[IntelligenceEvent]
    summary: Dict[str, Any]
    processing: Dict[str, Any]
