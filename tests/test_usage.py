"""
Tests for usage metering (tasks 5.1-5.3).

Covers:
- Billable event recorded on HTTP 200
- Non-billable event recorded on HTTP 4xx/5xx
- Monthly count aggregation
- GET /usage returns correct schema
- remaining = limit - used
"""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from api.auth.middleware import require_auth, _key_store
from api.auth.hashing import hash_api_key
from api.usage.meter import UsageMeter
from api.usage.router import router as usage_router, _meter, PLAN_LIMITS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_key_store():
    _key_store.clear()
    yield
    _key_store.clear()


def _register_key(key: str, plan: str, tenant_id: str = "t_test") -> str:
    hashed = hash_api_key(key)
    _key_store[hashed] = {
        "tenant_id": tenant_id,
        "plan": plan,
        "key_hash": hashed,
    }
    return key


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(usage_router)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# UsageMeter unit tests
# ---------------------------------------------------------------------------

class TestUsageMeter:
    def test_billable_on_200(self):
        meter = UsageMeter()
        meter.record("t1", "/extract", "passport", 120, 200)
        assert len(meter._records) == 1
        assert meter._records[0]["billable"] is True

    def test_non_billable_on_4xx(self):
        meter = UsageMeter()
        meter.record("t1", "/extract", "passport", 50, 400)
        assert meter._records[0]["billable"] is False

    def test_non_billable_on_5xx(self):
        meter = UsageMeter()
        meter.record("t1", "/extract", "passport", 50, 500)
        assert meter._records[0]["billable"] is False

    def test_record_stores_all_fields(self):
        meter = UsageMeter()
        meter.record("t2", "/extract-id", "id_card", 200, 200)
        r = meter._records[0]
        assert r["tenant_id"] == "t2"
        assert r["endpoint"] == "/extract-id"
        assert r["document_type"] == "id_card"
        assert r["response_time_ms"] == 200
        assert r["status_code"] == 200
        assert "created_at" in r

    def test_monthly_count_only_counts_billable(self):
        meter = UsageMeter()
        meter.record("t3", "/extract", "passport", 100, 200)
        meter.record("t3", "/extract", "passport", 100, 200)
        meter.record("t3", "/extract", "passport", 100, 422)  # non-billable

        from datetime import datetime, timezone
        period = datetime.now(timezone.utc).strftime("%Y-%m")
        count = meter.get_monthly_count("t3", period)
        assert count == 2

    def test_monthly_count_isolates_tenants(self):
        meter = UsageMeter()
        meter.record("t_a", "/extract", "passport", 100, 200)
        meter.record("t_b", "/extract", "passport", 100, 200)

        from datetime import datetime, timezone
        period = datetime.now(timezone.utc).strftime("%Y-%m")
        assert meter.get_monthly_count("t_a", period) == 1
        assert meter.get_monthly_count("t_b", period) == 1

    def test_monthly_count_wrong_period_returns_zero(self):
        meter = UsageMeter()
        meter.record("t4", "/extract", "passport", 100, 200)
        assert meter.get_monthly_count("t4", "1999-01") == 0

    def test_get_usage_summary_schema(self):
        meter = UsageMeter()
        summary = meter.get_usage_summary("t5", "starter", 500)
        assert "plan" in summary
        assert "period" in summary
        assert "extractions_used" in summary
        assert "extractions_limit" in summary
        assert "remaining" in summary

    def test_remaining_equals_limit_minus_used(self):
        meter = UsageMeter()
        meter.record("t6", "/extract", "passport", 100, 200)
        meter.record("t6", "/extract", "passport", 100, 200)
        summary = meter.get_usage_summary("t6", "starter", 500)
        assert summary["extractions_used"] == 2
        assert summary["remaining"] == 498

    def test_remaining_never_negative(self):
        meter = UsageMeter()
        # Record more than the limit
        for _ in range(25):
            meter.record("t7", "/extract", "passport", 100, 200)
        summary = meter.get_usage_summary("t7", "trial", 20)
        assert summary["remaining"] == 0


# ---------------------------------------------------------------------------
# GET /usage HTTP integration tests
# ---------------------------------------------------------------------------

class TestUsageEndpoint:
    def test_usage_requires_auth(self, client):
        r = client.get("/usage")
        assert r.status_code == 401

    def test_usage_returns_correct_schema(self, client):
        key = _register_key("sk_usage_1", "starter")
        # Reset meter state for this tenant
        _meter._records.clear()

        r = client.get("/usage", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        body = r.json()
        assert "plan" in body
        assert "period" in body
        assert "extractions_used" in body
        assert "extractions_limit" in body
        assert "remaining" in body

    def test_usage_plan_matches_tenant(self, client):
        key = _register_key("sk_usage_2", "growth")
        _meter._records.clear()

        r = client.get("/usage", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        body = r.json()
        assert body["plan"] == "growth"
        assert body["extractions_limit"] == PLAN_LIMITS["growth"]

    def test_usage_remaining_equals_limit_minus_used(self, client):
        key = _register_key("sk_usage_3", "trial", tenant_id="t_usage_3")
        _meter._records.clear()

        # Manually inject two billable records for this tenant
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        _meter._records.append(
            {"tenant_id": "t_usage_3", "endpoint": "/extract", "document_type": "passport",
             "response_time_ms": 100, "status_code": 200, "billable": True, "created_at": now}
        )
        _meter._records.append(
            {"tenant_id": "t_usage_3", "endpoint": "/extract", "document_type": "passport",
             "response_time_ms": 100, "status_code": 200, "billable": True, "created_at": now}
        )

        r = client.get("/usage", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200
        body = r.json()
        assert body["extractions_used"] == 2
        assert body["remaining"] == body["extractions_limit"] - 2
