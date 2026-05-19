from __future__ import annotations
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NNM_", extra="ignore")

    env: str = "local"
    log_level: str = "INFO"

    db_url: str
    db_pool_size: int = 10
    db_max_overflow: int = 20

    s3_bucket: str
    s3_region: str = "ap-northeast-2"
    s3_prefix: str = ""

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_fp16: bool = False
    embedding_batch_size: int = 4
    embedding_max_tokens: int = 8192

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    storage_root: str = "var"
    pdf_threads: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
