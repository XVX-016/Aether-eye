"""add aoi_daily_counts table

Revision ID: 0002_aoi_daily_counts
Revises: 0001_stage2_monitoring
Create Date: 2026-03-13
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_aoi_daily_counts"
down_revision = "0001_stage2_monitoring"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "aoi_daily_counts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("aoi_id", sa.String(length=64), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("aoi_id", "date", "event_type", name="uq_aoi_daily_counts_aoi_date_event"),
    )


def downgrade() -> None:
    op.drop_table("aoi_daily_counts")
