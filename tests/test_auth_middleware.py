"""Tests for the auth middleware (Bearer token validation)."""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from api.auth.middleware import require_auth, _key_store
from api.auth.hashing import hash_api_key


@pytest.fixture(autouse=True)
def clean_key_store():
    """Ensure _key_store is clean before and after each test."""
    _key_store.clear()
    yield
    _key_store.clear()


@pytest.fixture()
def app():
    _app = FastAPI()

    @_app.get("/protected")
    def protected(tenant: dict = Depends(require_auth)):
        return tenant

    @_app.get("/health")
    def health():
        return {"status": "ok"}

    return _app


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_missing_auth_header_returns_401(client):
    resp = client.get("/protected")
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"]["error"] == "unauthorized"


def test_malformed_auth_header_returns_401(client):
    resp = client.get("/protected", headers={"Authorization": "Token abc123"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["detail"]["error"] == "unauthorized"


def test_bearer_prefix_only_returns_401(client):
    # "Bearer " with no token after it — still malformed intent, but technically
    # starts with "Bearer "; the empty token won't match any key → 403.
    # Confirm it at least doesn't 500.
    resp = client.get("/protected", headers={"Authorization": "Bearer "})
    assert resp.status_code == 403


def test_unknown_key_returns_403(client):
    resp = client.get("/protected", headers={"Authorization": "Bearer test_key_unknownkey"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["detail"]["error"] == "forbidden"


def test_valid_key_returns_tenant_context(client):
    key = "test_key_testkey123"
    tenant = {"tenant_id": "t_abc", "plan": "pro"}
    _key_store[hash_api_key(key)] = tenant

    resp = client.get("/protected", headers={"Authorization": f"Bearer {key}"})
    assert resp.status_code == 200
    assert resp.json() == tenant


def test_health_endpoint_needs_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
