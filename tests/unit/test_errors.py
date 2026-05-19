from __future__ import annotations
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nnm.errors import (
    NnmError, PaperNotFound, PdfExtractionError,
    EmbeddingFailure, S3FetchError, PublicationMappingMissed,
    register_exception_handlers,
)


def test_hierarchy():
    for cls in (PaperNotFound, PdfExtractionError, EmbeddingFailure,
                S3FetchError, PublicationMappingMissed):
        assert issubclass(cls, NnmError)


def test_status_codes():
    assert PaperNotFound().status_code == 404
    assert PdfExtractionError().status_code == 422
    assert EmbeddingFailure().status_code == 502
    assert S3FetchError().status_code == 502


def test_handler_returns_json():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise PaperNotFound("missing paper 7")

    r = TestClient(app).get("/boom")
    assert r.status_code == 404
    assert r.json()["code"] == "paper_not_found"
    assert "missing paper 7" in r.json()["message"]
