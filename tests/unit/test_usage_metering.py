"""
Unit tests for usage metering.

Covers:
- Billable event recorded on HTTP 200
- Non-billable event recorded on HTTP 4xx/5xx
- Monthly aggregation counts only billable events
"""

import pytest
from datetime import datetime, timezone

from api.usage.meter import UsageMeter


# ---------------------------------------------------------------------------
# Billable vs non-billable events
# ---------------------------------------------------------------------------

class TestBillableEvents:
    def test_http_200_is_billable(self):
        meter = UsageMeter()
        meter.record("t1", "/extract", "passport", 120, 200)
        assert len(meter._records) == 1
        assert meter._records[0]["billable"] is True

    def test_http_201_is_not_billable(self):
        # Only 200 is billable per spec
        meter = UsageMeter()
        meter.record("t1", "/extract", "passport", 120, 201)
        assert meter._records[0]["billable"] is False

    def test_http_400_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t2", "/extract", "passport", 50, 400)
        assert meter._records[0]["billable"] is False

    def test_http_401_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t2", "/extract", "passport", 50, 401)
        assert meter._records[0]["billable"] is False

    def test_http_403_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t2", "/extract", "passport", 50, 403)
        assert meter._records[0]["billable"] is False

    def test_http_404_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t2", "/extract", "passport", 50, 404)
        assert meter._records[0]["billable"] is False

    def test_http_422_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t2", "/extract", "passport", 50, 422)
        assert meter._records[0]["billable"] is False

    def test_http_429_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t2", "/extract", "passport", 50, 429)
        assert meter._records[0]["billable"] is False

    def test_http_500_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t3", "/extract", "passport", 50, 500)
        assert meter._records[0]["billable"] is False

    def test_http_502_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t3", "/extract", "passport", 50, 502)
        assert meter._records[0]["billable"] is False

    def test_http_503_is_not_billable(self):
        meter = UsageMeter()
        meter.record("t3", "/extract", "passport", 50, 503)
        assert meter._records[0]["billable"] is False


# ---------------------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------------------

class TestMonthlyAggregation:
    def test_monthly_count_only_counts_billable_events(self):
        meter = UsageMeter()
        meter.record("t_agg", "/extract", "passport", 100, 200)
        meter.record("t_agg", "/extract", "passport", 100, 200)
        meter.record("t_agg", "/extract", "passport", 100, 200)
        meter.record("t_agg", "/extract", "passport", 100, 422)  # non-billable
        meter.record("t_agg", "/extract", "passport", 100, 500)  # non-billable

        period = datetime.now(timezone.utc).strftime("%Y-%m")
        count = meter.get_monthly_count("t_agg", period)
        assert count == 3

    def test_monthly_count_isolates_tenants(self):
        meter = UsageMeter()
        meter.record("t_a", "/extract", "passport", 100, 200)
        meter.record("t_a", "/extract", "passport", 100, 200)
        meter.record("t_b", "/extract", "passport", 100, 200)

        period = datetime.now(timezone.utc).strftime("%Y-%m")
        assert meter.get_monthly_count("t_a", period) == 2
        assert meter.get_monthly_count("t_b", period) == 1

    def test_monthly_count_wrong_period_returns_zero(self):
        meter = UsageMeter()
        meter.record("t_period", "/extract", "passport", 100, 200)
        assert meter.get_monthly_count("t_period", "1999-01") == 0

    def test_monthly_count_ignores_non_billable(self):
        meter = UsageMeter()
        # Record only non-billable events
        meter.record("t_nb", "/extract", "passport", 100, 400)
        meter.record("t_nb", "/extract", "passport", 100, 500)
        meter.record("t_nb", "/extract", "passport", 100, 422)

        period = datetime.now(timezone.utc).strftime("%Y-%m")
        count = meter.get_monthly_count("t_nb", period)
        assert count == 0


# ---------------------------------------------------------------------------
# Record storage
# ---------------------------------------------------------------------------

class TestRecordStorage:
    def test_record_stores_all_fields(self):
        meter = UsageMeter()
        meter.record("t_fields", "/extract-id", "id_card", 200, 200)
        r = meter._records[0]
        assert r["tenant_id"] == "t_fields"
        assert r["endpoint"] == "/extract-id"
        assert r["document_type"] == "id_card"
        assert r["response_time_ms"] == 200
        assert r["status_code"] == 200
        assert "created_at" in r
        assert "billable" in r

    def test_record_includes_timestamp(self):
        meter = UsageMeter()
        meter.record("t_ts", "/extract", "passport", 100, 200)
        r = meter._records[0]
        assert "created_at" in r
        # Verify it's a valid ISO 8601 timestamp
        datetime.fromisoformat(r["created_at"])


# ---------------------------------------------------------------------------
# Usage summary
# ---------------------------------------------------------------------------

class TestUsageSummary:
    def test_get_usage_summary_schema(self):
        meter = UsageMeter()
        summary = meter.get_usage_summary("t_summary", "starter", 500)
        assert "plan" in summary
        assert "period" in summary
        assert "extractions_used" in summary
        assert "extractions_limit" in summary
        assert "remaining" in summary

    def test_remaining_equals_limit_minus_used(self):
        meter = UsageMeter()
        meter.record("t_rem", "/extract", "passport", 100, 200)
        meter.record("t_rem", "/extract", "passport", 100, 200)
        summary = meter.get_usage_summary("t_rem", "starter", 500)
        assert summary["extractions_used"] == 2
        assert summary["remaining"] == 498

    def test_remaining_never_negative(self):
        meter = UsageMeter()
        # Record more than the limit
        for _ in range(25):
            meter.record("t_neg", "/extract", "passport", 100, 200)
        summary = meter.get_usage_summary("t_neg", "trial", 20)
        assert summary["remaining"] == 0
        assert summary["extractions_used"] == 25

    def test_usage_summary_plan_matches(self):
        meter = UsageMeter()
        summary = meter.get_usage_summary("t_plan", "growth", 5000)
        assert summary["plan"] == "growth"
        assert summary["extractions_limit"] == 5000
