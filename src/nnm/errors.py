from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class NnmError(Exception):
    status_code: int = 500
    code: str = "internal_error"


class PaperNotFound(NnmError):
    status_code = 404
    code = "paper_not_found"


class PdfExtractionError(NnmError):
    status_code = 422
    code = "pdf_extraction_failed"


class OcrRequiredError(NnmError):
    """1차 텍스트 추출이 비어 있는데 OCR fallback 이 비활성인 경우."""
    status_code = 422
    code = "ocr_required"


class EmbeddingFailure(NnmError):
    status_code = 502
    code = "embedding_failed"


class S3FetchError(NnmError):
    status_code = 502
    code = "s3_fetch_failed"


class PublicationMappingMissed(NnmError):
    status_code = 404
    code = "publication_mapping_missed"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(NnmError)
    async def _handle(request: Request, exc: NnmError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": str(exc)},
        )
