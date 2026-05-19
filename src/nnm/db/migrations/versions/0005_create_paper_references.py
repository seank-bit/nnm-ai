"""create paper_references

Revision ID: 0005
Revises: 0004
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_references",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("paper_id", sa.BigInteger(),
                  sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.UniqueConstraint("paper_id", "seq"),
    )


def downgrade() -> None:
    op.drop_table("paper_references")
