"""add flight state activity tables

Revision ID: 0004_flight_states
Revises: 0003_intel_articles
Create Date: 2026-03-24 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0004_flight_states"
down_revision = "0003_intel_articles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "flight_states",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("icao24", sa.String(length=16), nullable=False),
        sa.Column("callsign", sa.String(length=32), nullable=True),
        sa.Column("origin_country", sa.String(length=64), nullable=True),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lon", sa.Float(), nullable=True),
        sa.Column("altitude_m", sa.Float(), nullable=True),
        sa.Column("velocity_ms", sa.Float(), nullable=True),
        sa.Column("heading", sa.Float(), nullable=True),
        sa.Column("on_ground", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("site_id", sa.String(length=64), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.UniqueConstraint("icao24", "timestamp", name="uq_flight_states_icao24_timestamp"),
    )
    op.create_index("ix_flight_states_icao24", "flight_states", ["icao24"])
    op.create_index("ix_flight_states_site_id", "flight_states", ["site_id"])
    op.create_index("ix_flight_states_timestamp", "flight_states", ["timestamp"])

    op.create_table(
        "flight_daily_counts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("site_id", sa.String(length=64), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("unique_aircraft", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.UniqueConstraint("site_id", "date", name="uq_flight_daily_counts_site_date"),
    )
    op.create_index("ix_flight_daily_counts_site_id", "flight_daily_counts", ["site_id"])


def downgrade() -> None:
    op.drop_index("ix_flight_daily_counts_site_id", table_name="flight_daily_counts")
    op.drop_table("flight_daily_counts")

    op.drop_index("ix_flight_states_timestamp", table_name="flight_states")
    op.drop_index("ix_flight_states_site_id", table_name="flight_states")
    op.drop_index("ix_flight_states_icao24", table_name="flight_states")
    op.drop_table("flight_states")
