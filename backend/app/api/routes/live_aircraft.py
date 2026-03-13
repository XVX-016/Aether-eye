from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.live_aircraft import LiveAircraftState
from app.services.adsb_service import get_aircraft_in_bbox


router = APIRouter(tags=["live-aircraft"])


@router.get("/live-aircraft", response_model=list[LiveAircraftState])
async def live_aircraft(
    bbox: str = Query(..., description="lat_min,lat_max,lon_min,lon_max"),
) -> list[LiveAircraftState]:
    try:
        lat_min, lat_max, lon_min, lon_max = [float(part.strip()) for part in bbox.split(",")]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="bbox must be lat_min,lat_max,lon_min,lon_max") from exc

    try:
        data = get_aircraft_in_bbox(lat_min, lat_max, lon_min, lon_max)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenSky request failed: {exc}") from exc
    return [LiveAircraftState(**item) for item in data]
