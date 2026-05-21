"""
Tests for billing endpoints and Stripe webhook handling.

All Stripe API calls are mocked — no real network requests are made.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.auth.middleware import _key_store
from api.auth.hashing import hash_api_key
import api.billing.stripe_config as stripe_config
import api.billing.webhooks as webhooks_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_stores():
    """Clear shared in-memory stores before/after each test."""
    _key_store.clear()
    webhooks_module._tenant_store.clear()
    stripe_config._client_instance = None
    yield
    _key_store.clear()
    webhooks_module._tenant_store.clear()
    stripe_config._client_instance = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def mock_stripe():
    """Return a MagicMock Stripe client and inject it."""
    mock = MagicMock()
    stripe_config.set_stripe_client(mock)
    return mock


def _seed_tenant(key: str, tenant: dict):
    _key_store[hash_api_key(key)] = tenant


SEED_KEY = "test_key_seedkey00000000000000000000"
SEED_TENANT = {
    "tenant_id": "t_test_1",
    "email": "user@example.com",
    "plan": "starter",
    "stripe_customer_id": "cus_test123",
}


# ---------------------------------------------------------------------------
# 7.2  POST /billing/checkout
# ---------------------------------------------------------------------------

class TestCheckout:
    def test_checkout_requires_auth(self, client):
        resp = client.post("/billing/checkout", json={"plan": "starter"})
        assert resp.status_code == 401

    def test_checkout_invalid_plan_returns_400(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        resp = client.post(
            "/billing/checkout",
            json={"plan": "enterprise_unknown"},
            headers={"Authorization": f"Bearer {SEED_KEY}"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_plan"

    def test_checkout_starter_returns_url(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        mock_stripe.create_checkout_session.return_value = MagicMock(
            url="https://checkout.stripe.com/pay/cs_test_starter"
        )
        resp = client.post(
            "/billing/checkout",
            json={"plan": "starter"},
            headers={"Authorization": f"Bearer {SEED_KEY}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "checkout_url" in body
        assert body["checkout_url"].startswith("https://checkout.stripe.com/")

    def test_checkout_growth_returns_url(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        mock_stripe.create_checkout_session.return_value = MagicMock(
            url="https://checkout.stripe.com/pay/cs_test_growth"
        )
        resp = client.post(
            "/billing/checkout",
            json={"plan": "growth"},
            headers={"Authorization": f"Bearer {SEED_KEY}"},
        )
        assert resp.status_code == 200
        assert "checkout_url" in resp.json()

    def test_checkout_passes_correct_price_id(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        mock_stripe.create_checkout_session.return_value = MagicMock(url="https://checkout.stripe.com/x")
        client.post(
            "/billing/checkout",
            json={"plan": "starter"},
            headers={"Authorization": f"Bearer {SEED_KEY}"},
        )
        call_kwargs = mock_stripe.create_checkout_session.call_args[1]
        price_id = call_kwargs["line_items"][0]["price"]
        assert price_id == stripe_config.PLAN_CONFIG["starter"]["price_id"]

    def test_checkout_stripe_error_returns_502(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        mock_stripe.create_checkout_session.side_effect = Exception("Stripe down")
        resp = client.post(
            "/billing/checkout",
            json={"plan": "starter"},
            headers={"Authorization": f"Bearer {SEED_KEY}"},
        )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 7.7  GET /billing/portal
# ---------------------------------------------------------------------------

class TestPortal:
    def test_portal_requires_auth(self, client):
        resp = client.get("/billing/portal", follow_redirects=False)
        assert resp.status_code == 401

    def test_portal_without_stripe_customer_returns_400(self, client, mock_stripe):
        tenant_no_customer = {"tenant_id": "t_2", "email": "x@x.com", "plan": "starter"}
        _seed_tenant(SEED_KEY, tenant_no_customer)
        resp = client.get(
            "/billing/portal",
            headers={"Authorization": f"Bearer {SEED_KEY}"},
            follow_redirects=False,
        )
        assert resp.status_code == 400

    def test_portal_redirects_to_stripe(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        mock_stripe.create_portal_session.return_value = MagicMock(
            url="https://billing.stripe.com/session/bps_test"
        )
        resp = client.get(
            "/billing/portal",
            headers={"Authorization": f"Bearer {SEED_KEY}"},
            follow_redirects=False,
        )
        assert resp.status_code == 302
        assert resp.headers["location"].startswith("https://billing.stripe.com/")

    def test_portal_stripe_error_returns_502(self, client, mock_stripe):
        _seed_tenant(SEED_KEY, SEED_TENANT)
        mock_stripe.create_portal_session.side_effect = Exception("Stripe down")
        resp = client.get(
            "/billing/portal",
            headers={"Authorization": f"Bearer {SEED_KEY}"},
            follow_redirects=False,
        )
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 7.3  POST /webhooks/stripe — signature verification
# ---------------------------------------------------------------------------

def _make_event(event_type: str, data_object: dict) -> dict:
    return {
        "type": event_type,
        "data": {"object": data_object},
    }


class TestWebhookSignature:
    def test_missing_signature_returns_400(self, client, mock_stripe):
        resp = client.post(
            "/webhooks/stripe",
            content=b'{"type":"test"}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_signature"

    def test_invalid_signature_returns_400(self, client, mock_stripe):
        mock_stripe.construct_event.side_effect = Exception("Invalid signature")
        resp = client.post(
            "/webhooks/stripe",
            content=b'{"type":"test"}',
            headers={
                "content-type": "application/json",
                "stripe-signature": "t=bad,v1=bad",
            },
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "invalid_signature"

    def test_valid_signature_returns_200(self, client, mock_stripe):
        event = _make_event("unknown.event", {})
        mock_stripe.construct_event.return_value = event
        resp = client.post(
            "/webhooks/stripe",
            content=b'{}',
            headers={
                "content-type": "application/json",
                "stripe-signature": "t=123,v1=abc",
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"received": True}


# ---------------------------------------------------------------------------
# 7.4  invoice.payment_succeeded
# ---------------------------------------------------------------------------

class TestPaymentSucceeded:
    def _setup_tenant(self, customer_id: str = "cus_abc"):
        """Seed a tenant in the webhook store with a pre-existing API key."""
        key = generate_key = "test_key_existingkey0000000000000000"
        key_hash = hash_api_key(key)
        webhooks_module._tenant_store[customer_id] = {
            "tenant_id": "t_webhook_1",
            "email": "pay@example.com",
            "plan": "starter",
            "status": "active",
            "extractions_used": 250,
            "extractions_limit": 500,
            "api_key_hash": key_hash,
        }
        _key_store[key_hash] = {"tenant_id": "t_webhook_1"}
        return customer_id, key_hash

    def test_payment_succeeded_resets_counter(self, client, mock_stripe):
        customer_id, _ = self._setup_tenant()
        event = _make_event("invoice.payment_succeeded", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        resp = client.post(
            "/webhooks/stripe",
            content=b'{}',
            headers={"stripe-signature": "t=1,v1=x"},
        )
        assert resp.status_code == 200
        assert webhooks_module._tenant_store[customer_id]["extractions_used"] == 0

    def test_payment_succeeded_activates_key(self, client, mock_stripe):
        customer_id, key_hash = self._setup_tenant()
        # Remove key from store to simulate deactivated state
        _key_store.pop(key_hash, None)

        event = _make_event("invoice.payment_succeeded", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        # Key should be re-activated
        assert key_hash in _key_store

    def test_payment_succeeded_unknown_customer_is_noop(self, client, mock_stripe):
        event = _make_event("invoice.payment_succeeded", {"customer": "cus_unknown"})
        mock_stripe.construct_event.return_value = event

        resp = client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})
        assert resp.status_code == 200  # no error, just ignored


# ---------------------------------------------------------------------------
# 7.5  customer.subscription.deleted
# ---------------------------------------------------------------------------

class TestSubscriptionDeleted:
    def _setup_tenant(self, customer_id: str = "cus_del"):
        key = "test_key_activekey000000000000000000"
        key_hash = hash_api_key(key)
        webhooks_module._tenant_store[customer_id] = {
            "tenant_id": "t_del_1",
            "email": "cancel@example.com",
            "plan": "growth",
            "status": "active",
            "extractions_used": 100,
            "extractions_limit": 5000,
            "api_key_hash": key_hash,
        }
        _key_store[key_hash] = {"tenant_id": "t_del_1"}
        return customer_id, key_hash

    def test_subscription_deleted_deactivates_key(self, client, mock_stripe):
        customer_id, key_hash = self._setup_tenant()
        event = _make_event("customer.subscription.deleted", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert key_hash not in _key_store

    def test_subscription_deleted_marks_cancelled(self, client, mock_stripe):
        customer_id, _ = self._setup_tenant()
        event = _make_event("customer.subscription.deleted", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert webhooks_module._tenant_store[customer_id]["status"] == "cancelled"

    def test_deactivated_key_returns_403_on_auth(self, client, mock_stripe):
        customer_id, key_hash = self._setup_tenant()
        # Retrieve the plaintext key that maps to key_hash
        # We know the key from _setup_tenant
        event = _make_event("customer.subscription.deleted", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        # The key should no longer authenticate
        deactivated_key = "test_key_activekey000000000000000000"
        resp = client.get("/keys/list", headers={"Authorization": f"Bearer {deactivated_key}"})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# 7.6  checkout.session.completed — auto-provision API key
# ---------------------------------------------------------------------------

class TestCheckoutCompleted:
    def test_checkout_completed_provisions_api_key(self, client, mock_stripe):
        session = {
            "customer": "cus_new_1",
            "customer_email": "new@example.com",
            "client_reference_id": "t_new_1",
            "metadata": {"plan": "starter", "tenant_id": "t_new_1"},
        }
        event = _make_event("checkout.session.completed", session)
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        tenant = webhooks_module._tenant_store.get("cus_new_1")
        assert tenant is not None
        assert tenant["status"] == "active"
        assert tenant["api_key_hash"] is not None
        assert tenant["api_key_hash"] in _key_store

    def test_checkout_completed_key_is_usable(self, client, mock_stripe):
        session = {
            "customer": "cus_new_2",
            "customer_email": "new2@example.com",
            "client_reference_id": "t_new_2",
            "metadata": {"plan": "growth", "tenant_id": "t_new_2"},
        }
        event = _make_event("checkout.session.completed", session)
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        # The provisioned key should be in _key_store and usable
        tenant = webhooks_module._tenant_store["cus_new_2"]
        key_hash = tenant["api_key_hash"]
        assert key_hash in _key_store
        assert _key_store[key_hash]["tenant_id"] == "t_new_2"
