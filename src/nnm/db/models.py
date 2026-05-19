from __future__ import annotations
import datetime as dt
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger, CHAR, DateTime, ForeignKey, Index, Integer,
    PrimaryKeyConstraint, SmallInteger, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nnm.db.base import Base


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    s3_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    external_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    abstract: Mapped[str | None] = mapped_column(Text)
    venue: Mapped[str | None] = mapped_column(Text)
    published_year: Mapped[int | None] = mapped_column(SmallInteger)
    language: Mapped[str | None] = mapped_column(String(8))
    page_count: Mapped[int | None] = mapped_column(Integer)
    raw_json_path: Mapped[str | None] = mapped_column(Text)
    raw_md_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )

    chunks: Mapped[list[Chunk]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    references_: Mapped[list[PaperReference]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )


Index("papers_status_idx", Paper.status)
Index("papers_year_idx", Paper.published_year)
Index("papers_language_idx", Paper.language)
Index(
    "papers_external_id_idx", Paper.external_id,
    unique=True, postgresql_where=Paper.external_id.isnot(None),
)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("paper_id", "seq"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    section_level: Mapped[int | None] = mapped_column(SmallInteger)
    page_from: Mapped[int | None] = mapped_column(Integer)
    page_to: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_for_embed: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(8))

    paper: Mapped[Paper] = relationship(back_populates="chunks")
    embedding: Mapped[ChunkEmbedding | None] = relationship(
        back_populates="chunk", uselist=False, cascade="all, delete-orphan"
    )


Index("chunks_paper_idx", Chunk.paper_id)
Index("chunks_language_idx", Chunk.language)


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (PrimaryKeyConstraint("chunk_id", "model_name", "model_version"),)

    chunk_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    dense: Mapped[Any] = mapped_column(Vector(1024), nullable=False)
    sparse: Mapped[dict | None] = mapped_column(JSONB)
    colbert_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunk: Mapped[Chunk] = relationship(back_populates="embedding")


class PaperReference(Base):
    __tablename__ = "paper_references"
    __table_args__ = (UniqueConstraint("paper_id", "seq"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    paper: Mapped[Paper] = relationship(back_populates="references_")


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_name: Mapped[str] = mapped_column(Text, nullable=False)
    s3_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    total_files: Mapped[int | None] = mapped_column(Integer)
    processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    failed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    started_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class IngestJobItem(Base):
    __tablename__ = "ingest_job_items"
    __table_args__ = (PrimaryKeyConstraint("job_id", "s3_key"),)

    job_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ingest_jobs.id", ondelete="CASCADE"), nullable=False
    )
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    paper_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("papers.id"))
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )


Index("ingest_items_status_idx", IngestJobItem.job_id, IngestJobItem.status)


# 운영 read-only — alembic 마이그레이션에서 제외 (env.py include_object 필터)

class Publication(Base):
    __tablename__ = "publications"
    __table_args__ = {"info": {"managed_by": "operational"}}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)


class PublicationFile(Base):
    __tablename__ = "publication_files"
    __table_args__ = {"info": {"managed_by": "operational"}}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    publication_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("publications.id"), nullable=False
    )
