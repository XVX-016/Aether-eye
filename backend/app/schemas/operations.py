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
    today_flights: int = 0
    flight_baseline: float = 0.0
    flight_anomaly: str = "normal"


class FlightStateResponse(BaseModel):
    site_id: str | None = None
    icao24: str
    callsign: str | None = None
    origin_country: str | None = None
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    velocity_ms: float | None = None
    heading: float | None = None
    on_ground: bool = False
    timestamp: datetime


class FlightActivityResponse(BaseModel):
    site_id: str
    recent_count: int
    unique_aircraft: int
    on_ground_count: int
    airborne_count: int
    latest_states: list[FlightStateResponse]


class IntelArticleResponse(BaseModel):
    title: str
    url: str
    source: str | None = None
    source_tier: int
    published_at: datetime | None = None
    site_id: str | None = None
