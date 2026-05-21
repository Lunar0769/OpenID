"""Tests for POST /keys/create endpoint."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.auth.middleware import _key_store
from api.auth.hashing import hash_api_key


@pytest.fixture(autouse=True)
def clean_key_store():
    _key_store.clear()
    yield
    _key_store.clear()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _seed_tenant(key: str, tenant: dict):
    """Helper: register a key in the store so require_auth passes."""
    _key_store[hash_api_key(key)] = tenant


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def test_create_key_without_auth_returns_401(client):
    resp = client.post("/keys/create")
    assert resp.status_code == 401


def test_create_key_with_invalid_key_returns_403(client):
    resp = client.post("/keys/create", headers={"Authorization": "Bearer test_key_badkey"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_create_key_returns_200_with_api_key_and_created_at(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    assert resp.status_code == 200

    body = resp.json()
    assert "api_key" in body
    assert "created_at" in body


def test_created_key_has_test_key_prefix(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    assert resp.json()["api_key"].startswith("test_key_")


def test_created_key_is_stored_as_hash(client):
    seed_key = "test_key_seedkey00000000000000000000"
    tenant = {"tenant_id": "t_1", "plan": "pro"}
    _seed_tenant(seed_key, tenant)

    resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    new_key = resp.json()["api_key"]

    # The plaintext key must NOT be in the store
    assert new_key not in _key_store
    # But its hash must be present
    assert hash_api_key(new_key) in _key_store


def test_new_key_can_authenticate(client):
    seed_key = "test_key_seedkey00000000000000000000"
    tenant = {"tenant_id": "t_1", "plan": "pro"}
    _seed_tenant(seed_key, tenant)

    new_key = client.post(
        "/keys/create", headers={"Authorization": f"Bearer {seed_key}"}
    ).json()["api_key"]

    # The newly created key should itself be usable to create another key
    resp = client.post("/keys/create", headers={"Authorization": f"Bearer {new_key}"})
    assert resp.status_code == 200


def test_each_call_returns_unique_key(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    keys = set()
    for _ in range(5):
        resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
        keys.add(resp.json()["api_key"])
        # re-seed so auth keeps working with the original key
        _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    assert len(keys) == 5


def test_created_at_is_iso_format(client):
    from datetime import datetime
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    created_at = resp.json()["created_at"]
    # Should parse without error
    dt = datetime.fromisoformat(created_at)
    assert dt.tzinfo is not None  # must be timezone-aware (UTC)
