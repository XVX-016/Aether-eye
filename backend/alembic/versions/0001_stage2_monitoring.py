"""stage2 monitoring schema

Revision ID: 0001_stage2_monitoring
Revises: None
Create Date: 2026-03-11
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry


revision = "0001_stage2_monitoring"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "aoi_registry",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("aoi_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("polygon", Geometry("POLYGON", srid=4326), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("scan_frequency_hrs", sa.Integer(), nullable=False, server_default="6"),
        sa.Column("cloud_threshold", sa.Float(), nullable=False, server_default="20"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_aoi_registry_aoi_id", "aoi_registry", ["aoi_id"], unique=True)

    op.create_table(
        "satellite_scenes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("scene_id", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="copernicus-dataspace"),
        sa.Column("collection", sa.String(), nullable=False, server_default="sentinel-2-l2a"),
        sa.Column("aoi_id", sa.String(), sa.ForeignKey("aoi_registry.aoi_id"), nullable=True),
        sa.Column("aoi_name", sa.String(), nullable=True),
        sa.Column("datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("footprint", Geometry("POLYGON", srid=4326), nullable=True),
        sa.Column("cloud_cover", sa.Float(), nullable=True),
        sa.Column("asset_href", sa.Text(), nullable=True),
        sa.Column("geotiff_path", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="DISCOVERED"),
        sa.Column("processed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_satellite_scenes_scene_id", "satellite_scenes", ["scene_id"], unique=True)
    op.create_index("ix_satellite_scenes_status", "satellite_scenes", ["status"], unique=False)
    op.create_index("ix_satellite_scenes_aoi_id", "satellite_scenes", ["aoi_id"], unique=False)

    op.create_table(
        "tile_detections",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("scene_id", sa.String(), sa.ForeignKey("satellite_scenes.scene_id"), nullable=False),
        sa.Column("tile_x", sa.Integer(), nullable=False),
        sa.Column("tile_y", sa.Integer(), nullable=False),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("location", Geometry("POINT", srid=4326), nullable=False),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("change_score", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("detection_class", sa.String(), nullable=True),
        sa.Column("bbox", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "object_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("scene_id", sa.String(), sa.ForeignKey("satellite_scenes.scene_id"), nullable=True),
        sa.Column("lat", sa.Float(), nullable=False),
        sa.Column("lon", sa.Float(), nullable=False),
        sa.Column("location", Geometry("POINT", srid=4326), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("priority", sa.String(), nullable=False, server_default="MEDIUM"),
        sa.Column("detection_class", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_object_events_event_id", "object_events", ["event_id"], unique=True)
    op.create_index("ix_object_events_event_type", "object_events", ["event_type"], unique=False)

    op.create_table(
        "activity_alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("alert_id", sa.String(), nullable=False),
        sa.Column("alert_type", sa.String(), nullable=False),
        sa.Column("location_name", sa.String(), nullable=True),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lon", sa.Float(), nullable=True),
        sa.Column("location", Geometry("POINT", srid=4326), nullable=True),
        sa.Column("severity", sa.String(), nullable=False, server_default="MEDIUM"),
        sa.Column("triggered_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("tile_id", sa.String(), nullable=True),
        sa.Column("event_type", sa.String(), nullable=True),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("previous_count", sa.Integer(), nullable=True),
        sa.Column("current_count", sa.Integer(), nullable=True),
        sa.Column("delta", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_activity_alerts_alert_id", "activity_alerts", ["alert_id"], unique=True)

    op.create_table(
        "ingestion_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("aoi_id", sa.String(), nullable=False),
        sa.Column("last_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_ingestion_state_aoi_id", "ingestion_state", ["aoi_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_ingestion_state_aoi_id", table_name="ingestion_state")
    op.drop_table("ingestion_state")
    op.drop_index("ix_activity_alerts_alert_id", table_name="activity_alerts")
    op.drop_table("activity_alerts")
    op.drop_index("ix_object_events_event_type", table_name="object_events")
    op.drop_index("ix_object_events_event_id", table_name="object_events")
    op.drop_table("object_events")
    op.drop_table("tile_detections")
    op.drop_index("ix_satellite_scenes_aoi_id", table_name="satellite_scenes")
    op.drop_index("ix_satellite_scenes_status", table_name="satellite_scenes")
    op.drop_index("ix_satellite_scenes_scene_id", table_name="satellite_scenes")
    op.drop_table("satellite_scenes")
    op.drop_index("ix_aoi_registry_aoi_id", table_name="aoi_registry")
    op.drop_table("aoi_registry")
