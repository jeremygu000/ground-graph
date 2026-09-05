"""Unit tests for S3/MinIO object store adapter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from groundgraph.infrastructure.object_storage.s3_store import S3ObjectStore


class _FakeMinioClient:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_calls: list[tuple[str, str, bytes]] = []
        self.delete_calls: list[tuple[str, str]] = []

    def put_object(
        self,
        bucket: str,
        key: str,
        data: Any,
        length: int,
        content_type: str | None = None,
    ) -> None:
        content = data.read()
        self.objects[key] = content
        self.put_calls.append((bucket, key, content))

    def get_object(self, bucket: str, key: str) -> Any:
        class _Response:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

            def close(self) -> None:
                pass

            def release_conn(self) -> None:
                pass

        if key not in self.objects:
            raise KeyError("Object not found")
        return _Response(self.objects[key])

    def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop(key, None)
        self.delete_calls.append((bucket, key))

    def stat_object(self, bucket: str, key: str) -> Any:
        if key not in self.objects:
            raise KeyError("Object not found")
        return MagicMock()


class _FakeSettings:
    s3_endpoint_url = "http://localhost:9000"
    s3_access_key = "minioadmin"
    s3_secret_key = MagicMock(get_secret_value=lambda: "minioadmin")
    s3_region = "us-east-1"
    s3_use_ssl = False
    s3_bucket_raw = "raw"
    s3_bucket_processed = "processed"


@pytest.fixture
def fake_client() -> _FakeMinioClient:
    return _FakeMinioClient()


class TestS3ObjectStore:
    def setup_method(self) -> None:
        self.fake_settings = _FakeSettings()

    def test_put_and_get_raw(self, fake_client: _FakeMinioClient) -> None:
        store = S3ObjectStore.__new__(S3ObjectStore)
        store._client = fake_client
        store._raw_bucket = "raw"
        store._processed_bucket = "processed"

        async def run() -> None:
            await store.put_raw("key1", b"hello world", "text/plain")
            result = await store.get_raw("key1")
            assert result == b"hello world"

        asyncio.run(run())

    def test_exists_returns_true_when_object_present(self, fake_client: _FakeMinioClient) -> None:
        store = S3ObjectStore.__new__(S3ObjectStore)
        store._client = fake_client
        store._raw_bucket = "raw"
        store._processed_bucket = "processed"
        fake_client.objects["key1"] = b"data"

        async def run() -> None:
            assert await store.exists("key1") is True
            assert await store.exists("nonexistent") is False

        asyncio.run(run())

    def test_delete_removes_object(self, fake_client: _FakeMinioClient) -> None:
        store = S3ObjectStore.__new__(S3ObjectStore)
        store._client = fake_client
        store._raw_bucket = "raw"
        store._processed_bucket = "processed"
        fake_client.objects["key1"] = b"data"

        async def run() -> None:
            await store.delete_raw("key1")
            assert "key1" not in fake_client.objects
            assert len(fake_client.delete_calls) == 1

        asyncio.run(run())

    def test_object_key_format(self) -> None:
        store = S3ObjectStore.__new__(S3ObjectStore)
        key = store._object_key("namespace", uuid4(), "1")
        assert "namespace" in key
        assert "/v1" in key
