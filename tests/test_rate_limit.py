"""
Tests for rate limiting (task 4.1-4.5).

Covers:
- Per-minute rate limits (paid: 60, trial: 5)
- HTTP 429 with Retry-After header and correct JSON body
- Monthly quota: 90% warning, 100% blocked
- X-RateLimit-* headers on successful responses
- HTTP 402 on quota exceeded
"""

import time
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from api.auth.middleware import require_auth, _key_store
from api.auth.hashing import hash_api_key
from api.rate_limit.limiter import RateLimiter
from api.rate_limit.quota import QuotaChecker
from api.middleware.rate_limit_middleware import check_rate_limit_dep


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_key_store():
    _key_store.clear()
    yield
    _key_store.clear()


def _make_app(limiter: RateLimiter, quota: QuotaChecker | None = None):
    """Build a minimal FastAPI app wired to the given limiter/quota."""
    app = FastAPI()

    # Override the module-level limiter used by the dependency
    import api.middleware.rate_limit_middleware as mw
    mw._limiter = limiter

    @app.get("/extract")
    def extract(
        rate_info: dict = Depends(check_rate_limit_dep),
    ):
        # Simulate quota check if provided
        if quota is not None:
            tenant_id = "t_test"
            plan = "starter"
            limit = 500
            allowed, warning = quota.check_quota(tenant_id, plan, limit)
            if not allowed:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "quota_exceeded",
                        "message": "Monthly extraction quota reached. Upgrade your plan at https://openid.ai/upgrade.",
                    },
                )
            quota.record_usage(tenant_id)
            headers = {
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(rate_info["reset_ts"]),
            }
            if warning:
                headers["X-Usage-Warning"] = warning
            return JSONResponse({"status": "ok"}, headers=headers)

        headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset_ts"]),
        }
        return JSONResponse({"status": "ok"}, headers=headers)

    return app


def _register_key(key: str, plan: str) -> str:
    hashed = hash_api_key(key)
    _key_store[hashed] = {"tenant_id": "t_test", "plan": plan, "key_hash": hashed}
    return key


# ---------------------------------------------------------------------------
# RateLimiter unit tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_paid_plan_60th_request_allowed(self):
        limiter = RateLimiter()
        for i in range(60):
            allowed, limit, remaining, _ = limiter.check_rate_limit("hash_paid", "starter")
            assert allowed, f"Request {i+1} should be allowed"
        assert limit == 60

    def test_paid_plan_61st_request_denied(self):
        limiter = RateLimiter()
        for _ in range(60):
            limiter.check_rate_limit("hash_paid2", "starter")
        allowed, _, remaining, _ = limiter.check_rate_limit("hash_paid2", "starter")
        assert not allowed
        assert remaining == 0

    def test_trial_plan_5th_request_allowed(self):
        limiter = RateLimiter()
        for i in range(5):
            allowed, limit, _, _ = limiter.check_rate_limit("hash_trial", "trial")
            assert allowed, f"Request {i+1} should be allowed"
        assert limit == 5

    def test_trial_plan_6th_request_denied(self):
        limiter = RateLimiter()
        for _ in range(5):
            limiter.check_rate_limit("hash_trial2", "trial")
        allowed, _, remaining, _ = limiter.check_rate_limit("hash_trial2", "trial")
        assert not allowed
        assert remaining == 0

    def test_remaining_decrements(self):
        limiter = RateLimiter()
        _, limit, remaining, _ = limiter.check_rate_limit("hash_dec", "starter")
        assert remaining == limit - 1

    def test_reset_ts_is_future(self):
        limiter = RateLimiter()
        _, _, _, reset_ts = limiter.check_rate_limit("hash_ts", "starter")
        assert reset_ts > int(time.time())

    def test_different_keys_are_independent(self):
        limiter = RateLimiter()
        for _ in range(60):
            limiter.check_rate_limit("key_a", "starter")
        # key_a is exhausted; key_b should still be allowed
        allowed, _, _, _ = limiter.check_rate_limit("key_b", "starter")
        assert allowed


# ---------------------------------------------------------------------------
# QuotaChecker unit tests
# ---------------------------------------------------------------------------

class TestQuotaChecker:
    def test_within_quota_no_warning(self):
        q = QuotaChecker()
        allowed, warning = q.check_quota("t1", "starter", 500)
        assert allowed
        assert warning is None

    def test_approaching_limit_warning_at_90_percent(self):
        q = QuotaChecker()
        # Use 450 out of 500 (90%)
        for _ in range(450):
            q.record_usage("t2")
        allowed, warning = q.check_quota("t2", "starter", 500)
        assert allowed
        assert warning == "approaching_limit"

    def test_quota_exceeded_returns_false(self):
        q = QuotaChecker()
        for _ in range(500):
            q.record_usage("t3")
        allowed, warning = q.check_quota("t3", "starter", 500)
        assert not allowed
        assert warning is None

    def test_record_usage_increments(self):
        q = QuotaChecker()
        q.record_usage("t4")
        q.record_usage("t4")
        assert q._usage["t4"] == 2

    def test_just_below_90_percent_no_warning(self):
        q = QuotaChecker()
        # 449 out of 500 = 89.8% — no warning yet
        for _ in range(449):
            q.record_usage("t5")
        allowed, warning = q.check_quota("t5", "starter", 500)
        assert allowed
        assert warning is None


# ---------------------------------------------------------------------------
# HTTP integration tests (via TestClient)
# ---------------------------------------------------------------------------

class TestRateLimitHTTP:
    def test_429_on_rate_limit_exceeded(self):
        limiter = RateLimiter()
        app = _make_app(limiter)
        client = TestClient(app, raise_server_exceptions=False)
        key = _register_key("test_key_paid", "starter")
        headers = {"Authorization": f"Bearer {key}"}

        for _ in range(60):
            r = client.get("/extract", headers=headers)
            assert r.status_code == 200

        r = client.get("/extract", headers=headers)
        assert r.status_code == 429

    def test_429_body_and_retry_after_header(self):
        limiter = RateLimiter()
        app = _make_app(limiter)
        client = TestClient(app, raise_server_exceptions=False)
        key = _register_key("test_key_paid2", "starter")
        headers = {"Authorization": f"Bearer {key}"}

        for _ in range(60):
            client.get("/extract", headers=headers)

        r = client.get("/extract", headers=headers)
        assert r.status_code == 429
        body = r.json()
        assert body["detail"]["error"] == "rate_limit_exceeded"
        assert "retryAfter" in body["detail"]
        assert "Retry-After" in r.headers

    def test_trial_429_on_6th_request(self):
        limiter = RateLimiter()
        app = _make_app(limiter)
        client = TestClient(app, raise_server_exceptions=False)
        key = _register_key("sk_trial", "trial")
        headers = {"Authorization": f"Bearer {key}"}

        for _ in range(5):
            r = client.get("/extract", headers=headers)
            assert r.status_code == 200

        r = client.get("/extract", headers=headers)
        assert r.status_code == 429

    def test_x_ratelimit_headers_present(self):
        limiter = RateLimiter()
        app = _make_app(limiter)
        client = TestClient(app, raise_server_exceptions=False)
        key = _register_key("test_key_hdr", "starter")

        r = client.get("/extract", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        assert "X-RateLimit-Limit" in r.headers
        assert "X-RateLimit-Remaining" in r.headers
        assert "X-RateLimit-Reset" in r.headers
        assert r.headers["X-RateLimit-Limit"] == "60"

    def test_402_on_quota_exceeded(self):
        quota = QuotaChecker()
        for _ in range(500):
            quota.record_usage("t_test")

        limiter = RateLimiter()
        app = _make_app(limiter, quota=quota)
        client = TestClient(app, raise_server_exceptions=False)
        key = _register_key("test_key_quota", "starter")

        r = client.get("/extract", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 402
        body = r.json()
        assert body["error"] == "quota_exceeded"
        assert "upgrade" in body["message"].lower()

    def test_approaching_limit_header(self):
        quota = QuotaChecker()
        for _ in range(450):
            quota.record_usage("t_test")

        limiter = RateLimiter()
        app = _make_app(limiter, quota=quota)
        client = TestClient(app, raise_server_exceptions=False)
        key = _register_key("test_key_warn", "starter")

        r = client.get("/extract", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        assert r.headers.get("X-Usage-Warning") == "approaching_limit"
