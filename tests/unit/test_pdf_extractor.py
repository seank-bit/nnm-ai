from __future__ import annotations
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from nnm.services.pdf_extractor import PdfExtraction, PdfExtractor


@pytest.mark.asyncio
async def test_extract_writes_json_and_md(tmp_path: Path):
    ex = PdfExtractor(storage_root=tmp_path, threads=2)
    fake_json = {"metadata": {"title": "Sample"}, "elements": []}

    async def fake_run(pdf_path, out_dir, file_hash):
        out_dir.mkdir(parents=True, exist_ok=True)
        jp = out_dir / f"{file_hash}.json"
        mp = out_dir / f"{file_hash}.md"
        jp.write_text(json.dumps(fake_json), encoding="utf-8")
        mp.write_text("# Sample", encoding="utf-8")
        return jp, mp

    with patch.object(ex, "_run_opendataloader", new=AsyncMock(side_effect=fake_run)):
        result = await ex.extract(b"%PDF-1.4 dummy", file_hash="a" * 64)

    assert isinstance(result, PdfExtraction)
    assert result.json_doc["metadata"]["title"] == "Sample"
    assert result.markdown.startswith("# Sample")
