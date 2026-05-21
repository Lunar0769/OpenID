"""Tests for GET /keys/list endpoint."""

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


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def test_list_keys_without_auth_returns_401(client):
    resp = client.get("/keys/list")
    assert resp.status_code == 401


def test_list_keys_with_invalid_key_returns_403(client):
    resp = client.get("/keys/list", headers={"Authorization": "Bearer test_key_badkey"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_list_keys_returns_empty_when_no_keys_created(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    assert resp.status_code == 200
    assert resp.json() == {"keys": []}


def test_list_keys_returns_created_key(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})

    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    assert resp.status_code == 200
    keys = resp.json()["keys"]
    assert len(keys) == 1


def test_list_keys_entry_has_required_fields(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})

    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    entry = resp.json()["keys"][0]
    assert "id" in entry
    assert "last4" in entry
    assert "created_at" in entry


def test_list_keys_last4_is_exactly_4_chars(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    create_resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    new_key = create_resp.json()["api_key"]

    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    entry = resp.json()["keys"][0]
    assert len(entry["last4"]) == 4
    assert entry["last4"] == new_key[-4:]


def test_list_keys_id_is_not_full_key(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    create_resp = client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})
    new_key = create_resp.json()["api_key"]

    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    entry = resp.json()["keys"][0]
    assert entry["id"] != new_key


def test_list_keys_multiple_keys(client):
    seed_key = "test_key_seedkey00000000000000000000"
    _seed_tenant(seed_key, {"tenant_id": "t_1", "plan": "pro"})

    for _ in range(3):
        client.post("/keys/create", headers={"Authorization": f"Bearer {seed_key}"})

    resp = client.get("/keys/list", headers={"Authorization": f"Bearer {seed_key}"})
    assert len(resp.json()["keys"]) == 3


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def test_list_keys_only_returns_own_tenant_keys(client):
    key_t1 = "test_key_tenant1key0000000000000000"
    key_t2 = "test_key_tenant2key0000000000000000"
    _seed_tenant(key_t1, {"tenant_id": "t_1", "plan": "pro"})
    _seed_tenant(key_t2, {"tenant_id": "t_2", "plan": "free"})

    # t_1 creates a key
    client.post("/keys/create", headers={"Authorization": f"Bearer {key_t1}"})
    # t_2 creates a key
    client.post("/keys/create", headers={"Authorization": f"Bearer {key_t2}"})

    resp_t1 = client.get("/keys/list", headers={"Authorization": f"Bearer {key_t1}"})
    resp_t2 = client.get("/keys/list", headers={"Authorization": f"Bearer {key_t2}"})

    assert len(resp_t1.json()["keys"]) == 1
    assert len(resp_t2.json()["keys"]) == 1
