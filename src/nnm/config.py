from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="NNM_", extra="ignore",
        populate_by_name=True,
    )

    env: str = "local"
    log_level: str = "INFO"

    db_url: str
    db_pool_size: int = 10
    db_max_overflow: int = 20

    s3_bucket: str
    s3_region: str = "ap-northeast-2"
    s3_prefix: str = ""
    s3_pdf_bucket: str | None = None
    s3_pdf_prefix: str = ""
    # 추출 결과 (.json, .md) 보관 위치. 비우면 S3 업로드 안 함 (로컬 var/ 만).
    # bucket 미설정 시 s3_pdf_bucket → s3_bucket 순으로 fallback.
    s3_extracted_bucket: str | None = None
    s3_extracted_prefix: str = "newnonmuncom-extracted/"

    publication_csv: Path | None = None
    publication_csv_encoding: str = "cp949"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_fp16: bool = False
    embedding_batch_size: int = 4
    embedding_max_tokens: int = 8192
    embedding_colbert: bool = False
    embedding_sparse: bool = True

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    storage_root: str = "var"
    pdf_threads: int = 4
    # 1 PDF 추출 (opendataloader subprocess) 최대 허용 시간.
    # 초과 시 process kill → PdfExtractionError → BackfillService 가 skip 처리
    # + failed_pdfs_path 에 JSONL 한 줄 append (나중에 일괄 재처리용).
    pdf_extract_timeout_s: int = 1800
    # 추출 실패 (timeout / opendataloader non-zero exit / 산출물 누락) 한 PDF 를
    # 기록할 JSONL 파일. None 이면 기록 안 함. storage_root 기준 상대경로 가능.
    failed_pdfs_path: Path | None = Path("failed_pdfs.jsonl")

    # opendataloader-pdf hybrid 백엔드 (스캔 PDF OCR fallback).
    # 동작: 1차 텍스트 추출이 elements=0 일 때만 호출 (PdfExtractor.extract).
    # 텍스트 PDF 는 항상 1차에서 끝나므로 OCR 비용 안 듦.
    hybrid_url: str | None = None
    # OCR 재시도 시 hybrid 모드. full=모든 페이지 강제 백엔드 (스캔 PDF용 권장).
    hybrid_mode: str = "full"
    hybrid_timeout_ms: int = 180000
    # True 이면 1차 추출 elements=0 인 PDF (= OCR 필요) 를 skip.
    # ingest 만 빠르게 텍스트 PDF 위주로 처리하고 싶을 때 사용.
    pdf_skip_ocr: bool = True

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    rag_top_k: int = 5
    rag_max_context_chars: int = 8000
    rag_temperature: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
