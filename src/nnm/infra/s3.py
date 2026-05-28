from __future__ import annotations
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aioboto3
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from nnm.errors import S3FetchError

log = structlog.get_logger()


@dataclass
class S3Loader:
    bucket: str
    region: str
    _session: aioboto3.Session | None = None

    def _client_ctx(self):
        sess = self._session or aioboto3.Session()
        return sess.client("s3", region_name=self.region)

    async def _paginate(self, prefix: str) -> AsyncIterator[dict]:
        async with self._client_ctx() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                yield page

    async def list_keys(self, prefix: str) -> AsyncIterator[str]:
        async for page in self._paginate(prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    async def count_objects(self, prefix: str) -> int:
        total = 0
        async for page in self._paginate(prefix):
            total += page.get("KeyCount", 0)
        return total

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    async def _get_bytes(self, key: str) -> bytes:
        async with self._client_ctx() as s3:
            resp = await s3.get_object(Bucket=self.bucket, Key=key)
            return await resp["Body"].read()

    async def download(self, key: str) -> tuple[bytes, str]:
        try:
            data = await self._get_bytes(key)
        except Exception as e:
            raise S3FetchError(f"failed to fetch {key}: {e}") from e
        digest = hashlib.sha256(data).hexdigest()
        log.debug("s3.download", key=key, bytes=len(data))
        return data, digest

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
    async def upload_bytes(
        self, key: str, data: bytes, content_type: str = "application/octet-stream",
    ) -> None:
        async with self._client_ctx() as s3:
            await s3.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
            )
        log.debug("s3.upload", key=key, bytes=len(data), content_type=content_type)
