"""create vector and search indexes

Revision ID: 0007
Revises: 0006
"""
from __future__ import annotations
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunk_emb_dense_hnsw
        ON chunk_embeddings USING hnsw (dense vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunk_emb_sparse_gin
        ON chunk_embeddings USING gin (sparse jsonb_path_ops)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS chunks_text_gin
        ON chunks USING gin (to_tsvector('simple', text))
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_text_gin")
    op.execute("DROP INDEX IF EXISTS chunk_emb_sparse_gin")
    op.execute("DROP INDEX IF EXISTS chunk_emb_dense_hnsw")
