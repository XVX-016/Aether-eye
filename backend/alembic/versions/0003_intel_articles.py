"""add intel_articles table

Revision ID: 0003_intel_articles
Revises: 0002_aoi_daily_counts
Create Date: 2026-03-15 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0003_intel_articles"
down_revision = "0002_aoi_daily_counts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "intel_articles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("site_id", sa.String(length=64), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=True),
        sa.Column("source_tier", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text("now()")),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.UniqueConstraint("url", name="uq_intel_articles_url"),
    )
    op.create_index("ix_intel_articles_site_id", "intel_articles", ["site_id"])


def downgrade() -> None:
    op.drop_index("ix_intel_articles_site_id", table_name="intel_articles")
    op.drop_table("intel_articles")
