"""Tests for the JWT-based identity verification."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from groundgraph.api import dependencies
from groundgraph.application.settings import Settings


class TestGetIdentityLocalMode:
    """Tests for auth_mode=local (development defaults)."""

    def test_local_mode_returns_default_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In local mode, get_identity returns settings defaults, ignoring headers."""
        mock_settings = Settings(
            auth_mode="local",
            auth_default_tenant="dev-tenant",
            auth_default_principal="dev-user",
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        identity = dependencies.get_identity(
            request=MagicMock(),
            x_tenant_id="should-be-ignored",
            x_principal="should-be-ignored",
        )

        assert identity.tenant_id == "dev-tenant"
        assert identity.principal == "dev-user"

    def test_local_mode_without_headers_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local mode works even when no auth headers are provided."""
        mock_settings = Settings(auth_mode="local")
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        identity = dependencies.get_identity(
            request=MagicMock(),
            x_tenant_id=None,
            x_principal=None,
        )

        assert identity.tenant_id == "default"
        assert identity.principal == "engineering"


class TestGetIdentityHeaderMode:
    """Tests for auth_mode=header."""

    def test_header_mode_requires_trusted_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In header mode without auth_trusted_headers, requests are rejected."""
        mock_settings = Settings(
            auth_mode="header",
            auth_trusted_headers=False,
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            dependencies.get_identity(
                request=MagicMock(),
                x_tenant_id="tenant-1",
                x_principal="user-1",
            )

        assert exc_info.value.status_code == 401
        assert "not trusted" in exc_info.value.detail

    def test_header_mode_with_trusted_headers_uses_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In header mode with auth_trusted_headers=True, headers are used as identity."""
        mock_settings = Settings(
            auth_mode="header",
            auth_trusted_headers=True,
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        identity = dependencies.get_identity(
            request=MagicMock(),
            x_tenant_id="tenant-abc",
            x_principal="user-xyz",
        )

        assert identity.tenant_id == "tenant-abc"
        assert identity.principal == "user-xyz"

    def test_header_mode_rejects_missing_tenant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In header mode, missing X-Tenant-ID returns 401."""
        mock_settings = Settings(
            auth_mode="header",
            auth_trusted_headers=True,
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            dependencies.get_identity(
                request=MagicMock(),
                x_tenant_id=None,
                x_principal="user-1",
            )

        assert exc_info.value.status_code == 401
        assert "X-Tenant-ID" in exc_info.value.detail

    def test_header_mode_rejects_missing_principal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In header mode, missing X-Principal returns 401."""
        mock_settings = Settings(
            auth_mode="header",
            auth_trusted_headers=True,
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            dependencies.get_identity(
                request=MagicMock(),
                x_tenant_id="tenant-1",
                x_principal=None,
            )

        assert exc_info.value.status_code == 401
        assert "X-Principal" in exc_info.value.detail


class TestGetIdentityOidcMode:
    """Tests for auth_mode=oidc (JWT verification)."""

    def test_oidc_mode_requires_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In oidc mode, missing bearer token returns 401."""
        mock_settings = Settings(
            auth_mode="oidc",
            auth_jwks_url="https://auth.example.com/.well-known/jwks.json",
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            dependencies.get_identity(request=MagicMock())

        assert exc_info.value.status_code == 401
        assert "Bearer token required" in exc_info.value.detail

    def test_oidc_mode_rejects_non_bearer_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In oidc mode, non-Bearer Authorization header returns 401."""
        mock_settings = Settings(
            auth_mode="oidc",
            auth_jwks_url="https://auth.example.com/.well-known/jwks.json",
        )
        monkeypatch.setattr(dependencies, "get_settings", lambda: mock_settings)

        with pytest.raises(HTTPException) as exc_info:
            dependencies.get_identity(
                request=MagicMock(),
                authorization="Basic dXNlcjpwYXNz",
            )

        assert exc_info.value.status_code == 401
        assert "Bearer token required" in exc_info.value.detail
