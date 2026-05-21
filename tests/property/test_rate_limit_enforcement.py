"""
Property-based tests for per-minute rate limit enforcement.

# Feature: openid-ocr-platform, Property 7: Per-Minute Rate Limiting
"""

import pytest
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from api.auth.middleware import require_auth, _key_store
from api.auth.hashing import hash_api_key
from api.rate_limit.limiter import RateLimiter, RATE_LIMITS
from api.middleware.rate_limit_middleware import check_rate_limit_dep
import api.middleware.rate_limit_middleware as mw_module


# Paid plans that have a 60 req/min limit
PAID_PLANS = [plan for plan, limit in RATE_LIMITS.items() if limit == 60]


def _make_test_app(limiter: RateLimiter) -> FastAPI:
    """Build a minimal FastAPI app wired to the given limiter."""
    app = FastAPI()
    mw_module._limiter = limiter

    @app.get("/extract")
    def extract(rate_info: dict = Depends(check_rate_limit_dep)):
        return JSONResponse({"status": "ok"})

    return app


def _register_key(key: str, plan: str, tenant_id: str = "t_prop") -> None:
    hashed = hash_api_key(key)
    _key_store[hashed] = {"tenant_id": tenant_id, "plan": plan, "key_hash": hashed}


@pytest.fixture(autouse=True)
def clean_key_store():
    _key_store.clear()
    yield
    _key_store.clear()


# Feature: openid-ocr-platform, Property 7: Per-Minute Rate Limiting
@given(plan=st.sampled_from(PAID_PLANS))
@settings(max_examples=100, deadline=None)
def test_61st_request_returns_429_for_paid_plans(plan: str):
    """Validates: Requirements 7.1 — Per-Minute Rate Limiting.

    For any paid plan, when 60 requests are made within a single minute,
    the 61st request SHALL return HTTP 429.
    """
    limiter = RateLimiter()
    app = _make_test_app(limiter)
    client = TestClient(app, raise_server_exceptions=False)

    api_key = f"test_key_prop_{plan}_test000000000000000"
    _register_key(api_key, plan)
    headers = {"Authorization": f"Bearer {api_key}"}

    limit = RATE_LIMITS[plan]

    # Make exactly `limit` requests — all should succeed
    for i in range(limit):
        resp = client.get("/extract", headers=headers)
        assert resp.status_code == 200, (
            f"Request {i+1}/{limit} for plan={plan} should be 200, got {resp.status_code}"
        )

    # The (limit+1)th request must be rejected
    resp = client.get("/extract", headers=headers)
    assert resp.status_code == 429, (
        f"Request {limit+1} for plan={plan} should be 429, got {resp.status_code}"
    )


@given(plan=st.sampled_from(PAID_PLANS))
@settings(max_examples=50, deadline=None)
def test_60th_request_is_allowed_for_paid_plans(plan: str):
    """Validates: Requirements 7.1 — Per-Minute Rate Limiting.

    The 60th request (exactly at the limit) SHALL be allowed (HTTP 200).
    """
    limiter = RateLimiter()
    app = _make_test_app(limiter)
    client = TestClient(app, raise_server_exceptions=False)

    api_key = f"test_key_prop60_{plan}_test0000000000000"
    _register_key(api_key, plan)
    headers = {"Authorization": f"Bearer {api_key}"}

    limit = RATE_LIMITS[plan]

    for i in range(limit - 1):
        client.get("/extract", headers=headers)

    # The limit-th request must still be allowed
    resp = client.get("/extract", headers=headers)
    assert resp.status_code == 200, (
        f"The {limit}th request for plan={plan} should be 200, got {resp.status_code}"
    )
