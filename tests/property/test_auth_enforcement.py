"""
Property-based tests for authentication enforcement on all protected endpoints.

# Feature: openid-ocr-platform, Property 3: Authentication Enforcement
"""

import string
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from api.main import app
from api.auth.middleware import _key_store
import api.billing.stripe_config as stripe_config
import api.billing.webhooks as webhooks_module


# Protected endpoints: (method, path)
PROTECTED_ENDPOINTS = [
    ("POST", "/extract"),
    ("POST", "/extract-id"),
    ("GET", "/usage"),
    ("POST", "/keys/create"),
    ("GET", "/keys/list"),
    ("DELETE", "/keys/revoke"),
    ("POST", "/billing/checkout"),
    ("GET", "/billing/portal"),
]


@pytest.fixture(autouse=True)
def reset_stores():
    _key_store.clear()
    webhooks_module._tenant_store.clear()
    stripe_config._client_instance = None
    yield
    _key_store.clear()
    webhooks_module._tenant_store.clear()
    stripe_config._client_instance = None


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


import string

# ASCII printable characters safe for HTTP headers (excluding control chars)
_ASCII_HEADER_CHARS = string.ascii_letters + string.digits + string.punctuation + " "

# Feature: openid-ocr-platform, Property 3: Authentication Enforcement
@given(
    fake_key=st.one_of(
        st.none(),
        st.text(min_size=0, max_size=5, alphabet=_ASCII_HEADER_CHARS),
        st.text(min_size=6, max_size=200, alphabet=_ASCII_HEADER_CHARS).filter(
            lambda s: not s.startswith("Bearer ")
        ),
    )
)
@settings(max_examples=100)
def test_protected_endpoints_require_auth_no_header(fake_key):
    """Validates: Requirements 3.1 — Authentication Enforcement.

    For any protected endpoint, a request without a valid
    Authorization: Bearer <api_key> header SHALL return HTTP 401.
    """
    client = TestClient(app, raise_server_exceptions=False)

    for method, path in PROTECTED_ENDPOINTS:
        if fake_key is None:
            headers = {}
        else:
            headers = {"Authorization": fake_key}

        if method == "POST":
            resp = client.post(path, headers=headers, json={})
        elif method == "DELETE":
            resp = client.request("DELETE", path, headers={**headers, "content-type": "application/json"}, data='{"key_id":"abc"}')
        else:
            resp = client.get(path, headers=headers)

        assert resp.status_code == 401, (
            f"Expected 401 for {method} {path} with auth={fake_key!r}, "
            f"got {resp.status_code}"
        )


@given(
    fake_bearer=st.text(
        min_size=1, max_size=200,
        alphabet=_ASCII_HEADER_CHARS,
    ).map(lambda s: f"Bearer {s}")
)
@settings(max_examples=50)
def test_protected_endpoints_invalid_bearer_returns_401_or_403(fake_bearer):
    """Validates: Requirements 3.1 — Authentication Enforcement.

    A Bearer token that doesn't match any registered key SHALL return
    HTTP 401 or 403 (not 200).
    """
    client = TestClient(app, raise_server_exceptions=False)
    # Ensure key store is empty so no token can match
    _key_store.clear()

    for method, path in PROTECTED_ENDPOINTS:
        headers = {"Authorization": fake_bearer}

        if method == "POST":
            resp = client.post(path, headers=headers, json={})
        elif method == "DELETE":
            resp = client.request("DELETE", path, headers={**headers, "content-type": "application/json"}, data='{"key_id":"abc"}')
        else:
            resp = client.get(path, headers=headers)

        assert resp.status_code in (401, 403), (
            f"Expected 401/403 for {method} {path} with auth={fake_bearer!r}, "
            f"got {resp.status_code}"
        )
