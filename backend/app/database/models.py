from __future__ import annotations

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, synonym
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from app.database.session import Base


class AOIRegistry(Base):
    __tablename__ = "aoi_registry"

    id = Column(Integer, primary_key=True, index=True)
    aoi_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    polygon = Column(Geometry("POLYGON", srid=4326), nullable=False)
    bbox = Column(JSON, nullable=True)
    scan_frequency_hrs = Column(Integer, nullable=False, default=6)
    cloud_threshold = Column(Float, nullable=False, default=20.0)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    scenes = relationship("SatelliteScene", back_populates="aoi", lazy="selectin")


class SatelliteScene(Base):
    __tablename__ = "satellite_scenes"

    id = Column(Integer, primary_key=True, index=True)
    scene_id = Column(String, unique=True, index=True, nullable=False)
    source = Column(String, index=True, nullable=False, default="copernicus-dataspace")
    collection = Column(String, index=True, nullable=False, default="sentinel-2-l2a")
    aoi_id = Column(String, ForeignKey("aoi_registry.aoi_id"), index=True, nullable=True)
    aoi_name = Column(String, index=True, nullable=True)
    datetime = Column(DateTime(timezone=True), index=True, nullable=False)
    bbox = Column(JSON, nullable=True)
    footprint = Column(Geometry("POLYGON", srid=4326), nullable=True)
    cloud_cover = Column(Float, nullable=True)
    asset_href = Column(Text, nullable=True)
    geotiff_path = Column(String, nullable=True)
    status = Column(String, default="DISCOVERED", index=True)
    processed = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)

    aoi = relationship("AOIRegistry", back_populates="scenes", lazy="joined")
    detections = relationship("TileDetection", back_populates="scene", lazy="selectin")
    events = relationship("ObjectEvent", back_populates="scene", lazy="selectin")

    # Backward-compatible alias used by existing routes/tasks.
    local_path = synonym("geotiff_path")


class TileDetection(Base):
    __tablename__ = "tile_detections"

    id = Column(Integer, primary_key=True, index=True)
    scene_id = Column(String, ForeignKey("satellite_scenes.scene_id"), index=True, nullable=False)
    tile_x = Column(Integer, nullable=False)
    tile_y = Column(Integer, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    location = Column(Geometry("POINT", srid=4326), nullable=False)
    model_type = Column(String, index=True, nullable=False)
    change_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    detection_class = Column(String, index=True, nullable=True)
    bbox = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    scene = relationship("SatelliteScene", back_populates="detections", lazy="joined")


class ObjectEvent(Base):
    __tablename__ = "object_events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True, nullable=False)
    type = Column("event_type", String, index=True, nullable=False)
    scene_id = Column(String, ForeignKey("satellite_scenes.scene_id"), index=True, nullable=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    location = Column(Geometry("POINT", srid=4326), nullable=False)
    confidence = Column(Float, nullable=True)
    priority = Column(String, default="MEDIUM")
    detection_class = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    scene = relationship("SatelliteScene", back_populates="events", lazy="joined")


class ActivityAlert(Base):
    __tablename__ = "activity_alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String, unique=True, index=True, nullable=False)
    alert_type = Column(String, index=True, nullable=False)
    location_name = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    location = Column(Geometry("POINT", srid=4326), nullable=True)
    severity = Column(String, nullable=False, default="MEDIUM")
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    payload = Column(JSON, nullable=True)

    # Compatibility fields for existing activity endpoints/aggregator.
    tile_id = Column(String, index=True, nullable=True)
    event_type = Column(String, index=True, nullable=True)
    window_start = Column(DateTime(timezone=True), index=True, nullable=True)
    window_end = Column(DateTime(timezone=True), index=True, nullable=True)
    previous_count = Column(Integer, nullable=True)
    current_count = Column(Integer, nullable=True)
    delta = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AoiDailyCount(Base):
    __tablename__ = "aoi_daily_counts"
    __table_args__ = (
        UniqueConstraint("aoi_id", "date", "event_type", name="uq_aoi_daily_counts_aoi_date_event"),
    )

    id = Column(Integer, primary_key=True, index=True)
    aoi_id = Column(String(64), nullable=False, index=True)
    date = Column(Date, nullable=False)
    event_type = Column(String(64), nullable=False, index=True)
    count = Column(Integer, nullable=False, default=0, server_default="0")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class IntelArticle(Base):
    __tablename__ = "intel_articles"

    id = Column(Integer, primary_key=True)
    site_id = Column(String(64), nullable=True, index=True)
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=False, unique=True)
    source = Column(String(128), nullable=True)
    source_tier = Column(Integer, nullable=False, default=2, server_default="2")
    published_at = Column(DateTime(timezone=True), nullable=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
    summary = Column(Text, nullable=True)


class FlightState(Base):
    __tablename__ = "flight_states"
    __table_args__ = (
        UniqueConstraint("icao24", "timestamp", name="uq_flight_states_icao24_timestamp"),
    )

    id = Column(Integer, primary_key=True)
    icao24 = Column(String(16), nullable=False, index=True)
    callsign = Column(String(32), nullable=True)
    origin_country = Column(String(64), nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    altitude_m = Column(Float, nullable=True)
    velocity_ms = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    on_ground = Column(Boolean, nullable=False, default=False, server_default="false")
    site_id = Column(String(64), nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())


class FlightDailyCount(Base):
    __tablename__ = "flight_daily_counts"
    __table_args__ = (
        UniqueConstraint("site_id", "date", name="uq_flight_daily_counts_site_date"),
    )

    id = Column(Integer, primary_key=True)
    site_id = Column(String(64), nullable=False, index=True)
    date = Column(Date, nullable=False)
    count = Column(Integer, nullable=False, default=0, server_default="0")
    unique_aircraft = Column(Integer, nullable=False, default=0, server_default="0")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class IngestionState(Base):
    __tablename__ = "ingestion_state"

    id = Column(Integer, primary_key=True, index=True)
    aoi_id = Column(String, unique=True, index=True)
    last_timestamp = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Compatibility aliases while routes/services are migrated.
IntelligenceEvent = ObjectEvent
AircraftActivityEvent = ActivityAlert
