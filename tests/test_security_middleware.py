"""Tests for HSTS and HTTPS redirect security middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.security_middleware import HSTSMiddleware, HTTPSRedirectMiddleware


@pytest.fixture()
def app():
    _app = FastAPI()

    _app.add_middleware(HSTSMiddleware)
    _app.add_middleware(HTTPSRedirectMiddleware)

    @_app.get("/health")
    def health():
        return {"status": "ok"}

    return _app


@pytest.fixture()
def client(app):
    # allow_redirects=False so we can inspect 301 responses directly
    return TestClient(app, raise_server_exceptions=False, follow_redirects=False)


# ---------------------------------------------------------------------------
# HSTS header
# ---------------------------------------------------------------------------

def test_hsts_header_present_on_normal_response(client):
    resp = client.get("/health")
    assert "strict-transport-security" in resp.headers
    assert "max-age=31536000" in resp.headers["strict-transport-security"]
    assert "includeSubDomains" in resp.headers["strict-transport-security"]


def test_hsts_header_present_on_https_forwarded_request(client):
    resp = client.get("/health", headers={"X-Forwarded-Proto": "https"})
    assert "strict-transport-security" in resp.headers
    assert "max-age=31536000" in resp.headers["strict-transport-security"]


# ---------------------------------------------------------------------------
# HTTPS redirect
# ---------------------------------------------------------------------------

def test_http_forwarded_proto_returns_301(client):
    resp = client.get(
        "http://testserver/health",
        headers={"X-Forwarded-Proto": "http"},
    )
    assert resp.status_code == 301


def test_http_redirect_location_uses_https(client):
    resp = client.get(
        "http://testserver/health",
        headers={"X-Forwarded-Proto": "http"},
    )
    assert resp.status_code == 301
    location = resp.headers["location"]
    assert location.startswith("https://")


def test_https_forwarded_proto_passes_through(client):
    resp = client.get("/health", headers={"X-Forwarded-Proto": "https"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_no_forwarded_proto_passes_through(client):
    """Local dev requests without X-Forwarded-Proto should not be redirected."""
    resp = client.get("/health")
    assert resp.status_code == 200
