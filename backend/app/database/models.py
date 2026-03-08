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
