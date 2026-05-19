"""create papers

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "papers",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("s3_key", sa.Text(), nullable=False, unique=True),
        sa.Column("file_hash", sa.CHAR(64), nullable=False),
        sa.Column("external_id", postgresql.UUID(as_uuid=True)),
        sa.Column("title", sa.Text()),
        sa.Column("authors", postgresql.ARRAY(sa.Text())),
        sa.Column("abstract", sa.Text()),
        sa.Column("venue", sa.Text()),
        sa.Column("published_year", sa.SmallInteger()),
        sa.Column("language", sa.String(8)),
        sa.Column("page_count", sa.Integer()),
        sa.Column("raw_json_path", sa.Text()),
        sa.Column("raw_md_path", sa.Text()),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
    )
    op.create_index("papers_status_idx", "papers", ["status"])
    op.create_index("papers_year_idx", "papers", ["published_year"])
    op.create_index("papers_language_idx", "papers", ["language"])
    op.create_index(
        "papers_external_id_idx", "papers", ["external_id"],
        unique=True, postgresql_where=sa.text("external_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_table("papers")
