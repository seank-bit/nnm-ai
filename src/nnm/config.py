from __future__ import annotations
from functools import lru_cache

from pydantic import Field
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

    ops_db_host: str | None = Field(default=None, validation_alias="DB_HOST")
    ops_db_port: int = Field(default=3306, validation_alias="DB_PORT")
    ops_db_database: str | None = Field(default=None, validation_alias="DB_DATABASE")
    ops_db_username: str | None = Field(default=None, validation_alias="DB_USERNAME")
    ops_db_password: str | None = Field(default=None, validation_alias="DB_PASSWORD")

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

    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    rag_top_k: int = 5
    rag_max_context_chars: int = 8000
    rag_temperature: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
