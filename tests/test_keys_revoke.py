"""Tests for DELETE /keys/revoke endpoint."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.auth.middleware import _key_store
from api.auth.hashing import hash_api_key
from api.keys.router import _key_registry


@pytest.fixture(autouse=True)
def clean_stores():
    _key_store.clear()
    _key_registry.clear()
    yield
    _key_store.clear()
    _key_registry.clear()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _seed_tenant(key: str, tenant: dict):
    _key_store[hash_api_key(key)] = tenant


def _delete_revoke(client, key_id: str, auth_key: str | None = None):
    """Helper to send DELETE /keys/revoke with a JSON body."""
    headers = {}
    if auth_key:
        headers["Authorization"] = f"Bearer {auth_key}"
    return client.request(
        "DELETE",
        "/keys/revoke",
        json={"key_id": key_id},
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def test_revoke_without_auth_returns_401(client):
    resp = _delete_revoke(client, "abc12345")
    assert resp.status_code == 401


def test_revoke_with_invalid_key_returns_403(client):
    resp = _delete_revoke(client, "abc12345", auth_key="test_key_badkey")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_revoke_key_returns_revoked_true(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    create_resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    assert create_resp.status_code == 200

    list_resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    key_id = list_resp.json()["keys"][0]["id"]

    revoke_resp = _delete_revoke(client, key_id, auth_key=seed_key)
    assert revoke_resp.status_code == 200
    assert revoke_resp.json() == {"revoked": True}


def test_revoked_key_no_longer_authenticates(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    create_resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    new_key = create_resp.json()["api_key"]

    list_resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    key_id = list_resp.json()["keys"][0]["id"]

    _delete_revoke(client, key_id, auth_key=seed_key)

    # The revoked key should no longer work
    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {new_key}"})
    assert resp.status_code == 403


def test_revoked_key_disappears_from_list(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})

    list_resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    key_id = list_resp.json()["keys"][0]["id"]

    _delete_revoke(client, key_id, auth_key=seed_key)

    list_after = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    assert list_after.json()["keys"] == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_revoke_nonexistent_key_returns_404(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    resp = _delete_revoke(client, "00000000", auth_key=seed_key)
    assert resp.status_code == 404


def test_revoke_other_tenants_key_returns_404(client):
    key_t1 = "test_key_tenant1key0000000000000000"
    key_t2 = "test_key_tenant2key0000000000000000"
    _seed_tenant(key_t1, {"tenant_id": "t_1", "plan": "pro"})
    _seed_tenant(key_t2, {"tenant_id": "t_2", "plan": "free"})

    # t_1 creates a key
    client.post("/keys/create", headers={"Authorization": f"Bearer {key_t1}"})
    list_resp = client.get("/keys/list", headers={"Authorization": f"Bearer {key_t1}"})
    key_id = list_resp.json()["keys"][0]["id"]

    # t_2 tries to revoke t_1's key
    resp = _delete_revoke(client, key_id, auth_key=key_t2)
    assert resp.status_code == 404
