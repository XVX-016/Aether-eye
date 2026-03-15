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


class AoiBaselineResponse(BaseModel):
    aoi_id: str
    event_type_baselines: dict[str, float]
    data_points: int
    lookback_days: int


class SiteStatusResponse(BaseModel):
    id: str
    name: str
    type: str
    priority: str
    country: str
    today_count: int | None = None
    baseline: float | None = None
    anomaly_factor: float | None = None
    status: str
