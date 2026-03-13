from __future__ import annotations

from pydantic import BaseModel, Field


class LiveAircraftState(BaseModel):
    icao24: str = Field(..., min_length=1)
    callsign: str | None = None
    origin_country: str | None = None
    lat: float | None = None
    lon: float | None = None
    altitude: float | None = None
    velocity: float | None = None
    heading: float | None = None
    on_ground: bool | None = None
