"""create chunk_embeddings

Revision ID: 0004
Revises: 0003
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.BigInteger(),
                  sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("dense", Vector(1024), nullable=False),
        sa.Column("sparse", postgresql.JSONB()),
        sa.Column("colbert_path", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("chunk_id", "model_name", "model_version"),
    )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
