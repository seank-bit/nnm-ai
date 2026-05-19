from __future__ import annotations
import hashlib
import pytest
from unittest.mock import AsyncMock, patch

from nnm.infra.s3 import S3Loader


async def _async_iter(items):
    for x in items:
        yield x


@pytest.mark.asyncio
async def test_list_keys_paginates():
    loader = S3Loader(bucket="b", region="ap-northeast-2")
    fake_pages = [
        {"Contents": [{"Key": "papers/a.pdf"}, {"Key": "papers/b.pdf"}]},
        {"Contents": [{"Key": "papers/c.pdf"}]},
    ]
    with patch.object(loader, "_paginate", return_value=_async_iter(fake_pages)):
        keys = [k async for k in loader.list_keys("papers/")]
    assert keys == ["papers/a.pdf", "papers/b.pdf", "papers/c.pdf"]


@pytest.mark.asyncio
async def test_download_returns_bytes_and_sha256():
    loader = S3Loader(bucket="b", region="ap-northeast-2")
    body = b"%PDF-1.4 fake"
    expected = hashlib.sha256(body).hexdigest()
    with patch.object(loader, "_get_bytes", new=AsyncMock(return_value=body)):
        data, digest = await loader.download("papers/x.pdf")
    assert data == body
    assert digest == expected
