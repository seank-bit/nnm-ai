"""create ingest_jobs

Revision ID: 0006
Revises: 0005
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingest_jobs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("job_name", sa.Text(), nullable=False),
        sa.Column("s3_prefix", sa.Text(), nullable=False),
        sa.Column("total_files", sa.Integer()),
        sa.Column("processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "ingest_job_items",
        sa.Column("job_id", sa.BigInteger(),
                  sa.ForeignKey("ingest_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("s3_key", sa.Text(), nullable=False),
        sa.Column("paper_id", sa.BigInteger(), sa.ForeignKey("papers.id")),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text()),
        sa.Column("attempts", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("job_id", "s3_key"),
    )
    op.create_index("ingest_items_status_idx", "ingest_job_items", ["job_id", "status"])


def downgrade() -> None:
    op.drop_table("ingest_job_items")
    op.drop_table("ingest_jobs")
