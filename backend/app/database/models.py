from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database.session import Base

class IntelligenceEvent(Base):
    __tablename__ = "intelligence_events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True)
    type = Column(String, index=True)
    lat = Column(Float)
    lon = Column(Float)
    confidence = Column(Float)
    priority = Column(String, default="MEDIUM")
    metadata_json = Column(JSON, nullable=True) # For flexible aircraft classes, etc.
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SatelliteScene(Base):
    __tablename__ = "satellite_scenes"

    id = Column(Integer, primary_key=True, index=True)
    scene_id = Column(String, index=True)
    collection = Column(String, index=True)
    datetime = Column(DateTime(timezone=True), index=True)
    bbox = Column(JSON, nullable=True)
    cloud_cover = Column(Float, nullable=True)
    asset_href = Column(String)
    status = Column(String, default="NEW", index=True)
    local_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class IngestionState(Base):
    __tablename__ = "ingestion_state"

    id = Column(Integer, primary_key=True, index=True)
    aoi_id = Column(String, unique=True, index=True)
    last_timestamp = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AircraftActivityEvent(Base):
    __tablename__ = "aircraft_activity_events"

    id = Column(Integer, primary_key=True, index=True)
    tile_id = Column(String, index=True)
    event_type = Column(String, index=True)
    window_start = Column(DateTime(timezone=True), index=True)
    window_end = Column(DateTime(timezone=True), index=True)
    previous_count = Column(Integer)
    current_count = Column(Integer)
    delta = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
