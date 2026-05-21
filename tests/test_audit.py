"""
Tests for AuditLoggerMiddleware (tasks 8.1 – 8.4).

Covers:
- Every request gets a unique request_id (UUID v4)
- X-Request-ID header present in all responses
- 401 response flagged as security_event=True
- 403 response flagged as security_event=True
- 200 response has security_event=False
- Audit record contains no PII fields
- response_time_ms is a non-negative integer
"""

import uuid
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.middleware.audit_middleware import AuditLoggerMiddleware, _PII_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(status_code: int = 200, audit_store: list | None = None):
    """Build a minimal app with AuditLoggerMiddleware and a single endpoint."""
    store = audit_store if audit_store is not None else []
    app = FastAPI()
    app.add_middleware(AuditLoggerMiddleware, audit_store=store)

    @app.get("/test")
    def endpoint():
        return JSONResponse(content={"ok": True}, status_code=status_code)

    return app, store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRequestId:
    def test_every_request_gets_unique_request_id(self):
        app, store = _make_app()
        client = TestClient(app, raise_server_exceptions=False)

        client.get("/test")
        client.get("/test")

        assert len(store) == 2
        id1 = store[0]["request_id"]
        id2 = store[1]["request_id"]
        assert id1 != id2

    def test_request_id_is_valid_uuid_v4(self):
        app, store = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")

        rid = store[0]["request_id"]
        parsed = uuid.UUID(rid)
        assert parsed.version == 4


class TestXRequestIDHeader:
    def test_x_request_id_header_present_on_200(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test")
        assert "x-request-id" in resp.headers

    def test_x_request_id_matches_audit_record(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test")
        assert resp.headers["x-request-id"] == store[0]["request_id"]

    def test_x_request_id_header_present_on_401(self):
        app, store = _make_app(401)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test")
        assert "x-request-id" in resp.headers

    def test_x_request_id_header_present_on_403(self):
        app, store = _make_app(403)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/test")
        assert "x-request-id" in resp.headers


class TestSecurityEvents:
    def test_401_flagged_as_security_event(self):
        app, store = _make_app(401)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["security_event"] is True

    def test_403_flagged_as_security_event(self):
        app, store = _make_app(403)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["security_event"] is True

    def test_200_not_a_security_event(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["security_event"] is False

    def test_429_not_a_security_event(self):
        app, store = _make_app(429)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["security_event"] is False


class TestNoPII:
    def test_audit_record_has_no_pii_fields(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")

        record = store[0]
        for pii_field in _PII_FIELDS:
            assert pii_field not in record, f"PII field '{pii_field}' found in audit record"

    def test_pii_constants_cover_required_fields(self):
        """Ensure the PII guard list includes all spec-required fields."""
        required = {"name", "document_number", "dob", "expiry", "country"}
        assert required.issubset(_PII_FIELDS)


class TestResponseTimeMs:
    def test_response_time_ms_is_non_negative_integer(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")

        rt = store[0]["response_time_ms"]
        assert isinstance(rt, int)
        assert rt >= 0


class TestAuditRecordFields:
    def test_required_fields_present(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")

        record = store[0]
        required = {
            "request_id", "timestamp", "endpoint", "http_method",
            "source_ip", "response_status", "response_time_ms",
            "security_event", "tenant_id", "api_key_hash",
        }
        for field in required:
            assert field in record, f"Required field '{field}' missing from audit record"

    def test_endpoint_recorded_correctly(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["endpoint"] == "/test"

    def test_http_method_recorded(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["http_method"] == "GET"

    def test_response_status_recorded(self):
        app, store = _make_app(200)
        client = TestClient(app, raise_server_exceptions=False)
        client.get("/test")
        assert store[0]["response_status"] == 200
