from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CountResponse(BaseModel):
    count: int


class OperationsEvent(BaseModel):
    event_id: str
    event_type: str
    lat: float | None = None
    lon: float | None = None
    confidence: float | None = None
    priority: str
    timestamp: datetime
    aoi_name: str | None = None
