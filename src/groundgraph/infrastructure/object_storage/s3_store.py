"""S3/MinIO object storage adapter.

Implements the ``ObjectStore`` port from ``application.ports``.
Uses the ``minio`` sync SDK wrapped with ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import io
from uuid import UUID

from minio import Minio
from minio.error import S3Error

from groundgraph.application.ports import ObjectStore
from groundgraph.application.settings import Settings


class S3ObjectStore(ObjectStore):
    def __init__(self, settings: Settings) -> None:
        self._client = Minio(
            endpoint=settings.s3_endpoint_url.removeprefix("http://").removeprefix("https://"),
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key.get_secret_value(),
            region=settings.s3_region,
            secure=settings.s3_use_ssl,
        )
        self._raw_bucket = settings.s3_bucket_raw
        self._processed_bucket = settings.s3_bucket_processed

    def _object_key(self, namespace: str, object_id: UUID, version: str | None = None) -> str:
        version_part = f"/v{version}" if version else ""
        return f"{namespace}/{object_id}{version_part}"

    async def put_raw(self, key: str, data: bytes, content_type: str | None = None) -> None:
        def _put() -> None:
            self._client.put_object(
                self._raw_bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type or "application/octet-stream",
            )

        await asyncio.to_thread(_put)

    async def get_raw(self, key: str) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(self._raw_bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(_get)

    async def delete_raw(self, key: str) -> None:
        def _del() -> None:
            self._client.remove_object(self._raw_bucket, key)

        await asyncio.to_thread(_del)

    async def exists(self, key: str) -> bool:
        def _exists() -> bool:
            try:
                self._client.stat_object(self._raw_bucket, key)
            except S3Error as e:
                if e.code in ("NoSuchKey", "NoSuchVersion"):
                    return False
                raise
            return True

        return await asyncio.to_thread(_exists)
