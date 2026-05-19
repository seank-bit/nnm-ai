"""create chunks

Revision ID: 0003
Revises: 0002
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("paper_id", sa.BigInteger(),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text()),
        sa.Column("section_level", sa.SmallInteger()),
        sa.Column("page_from", sa.Integer()),
        sa.Column("page_to", sa.Integer()),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_for_embed", sa.Text(), nullable=False),
        sa.Column("language", sa.String(8)),
        sa.UniqueConstraint("paper_id", "seq"),
    )
    op.create_index("chunks_paper_idx", "chunks", ["paper_id"])
    op.create_index("chunks_language_idx", "chunks", ["language"])


def downgrade() -> None:
    op.drop_table("chunks")
