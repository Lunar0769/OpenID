"""
Unit tests for Stripe webhook handlers.

Covers:
- invoice.payment_succeeded resets monthly counter
- customer.subscription.deleted deactivates API key
- Invalid signature returns 400
"""

import pytest
from unittest.mock import MagicMock
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
    mock = MagicMock()
    stripe_config.set_stripe_client(mock)
    return mock


def _make_event(event_type: str, data_object: dict) -> dict:
    return {"type": event_type, "data": {"object": data_object}}


def _seed_tenant(customer_id: str, key: str, extractions_used: int = 0) -> str:
    key_hash = hash_api_key(key)
    webhooks_module._tenant_store[customer_id] = {
        "tenant_id": f"t_{customer_id}",
        "email": f"{customer_id}@example.com",
        "plan": "starter",
        "status": "active",
        "extractions_used": extractions_used,
        "extractions_limit": 500,
        "api_key_hash": key_hash,
    }
    _key_store[key_hash] = {"tenant_id": f"t_{customer_id}"}
    return key_hash


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

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
        mock_stripe.construct_event.return_value = _make_event("unknown.event", {})
        resp = client.post(
            "/webhooks/stripe",
            content=b'{}',
            headers={"stripe-signature": "t=1,v1=abc"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"received": True}


# ---------------------------------------------------------------------------
# invoice.payment_succeeded
# ---------------------------------------------------------------------------

class TestPaymentSucceeded:
    def test_payment_succeeded_resets_monthly_counter(self, client, mock_stripe):
        key = "test_key_pay_test_000000000000000000"
        customer_id = "cus_pay_test"
        _seed_tenant(customer_id, key, extractions_used=250)

        event = _make_event("invoice.payment_succeeded", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        resp = client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert resp.status_code == 200
        assert webhooks_module._tenant_store[customer_id]["extractions_used"] == 0

    def test_payment_succeeded_keeps_status_active(self, client, mock_stripe):
        key = "test_key_pay_active_0000000000000000"
        customer_id = "cus_pay_active"
        _seed_tenant(customer_id, key)

        event = _make_event("invoice.payment_succeeded", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert webhooks_module._tenant_store[customer_id]["status"] == "active"

    def test_payment_succeeded_restores_deactivated_key(self, client, mock_stripe):
        key = "test_key_pay_restore_000000000000000"
        customer_id = "cus_pay_restore"
        key_hash = _seed_tenant(customer_id, key)
        # Simulate deactivated key
        _key_store.pop(key_hash, None)

        event = _make_event("invoice.payment_succeeded", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert key_hash in _key_store

    def test_payment_succeeded_unknown_customer_is_noop(self, client, mock_stripe):
        event = _make_event("invoice.payment_succeeded", {"customer": "cus_unknown_xyz"})
        mock_stripe.construct_event.return_value = event

        resp = client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# customer.subscription.deleted
# ---------------------------------------------------------------------------

class TestSubscriptionDeleted:
    def test_subscription_deleted_deactivates_api_key(self, client, mock_stripe):
        key = "test_key_sub_del_test_0000000000000"
        customer_id = "cus_sub_del"
        key_hash = _seed_tenant(customer_id, key)

        event = _make_event("customer.subscription.deleted", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert key_hash not in _key_store

    def test_subscription_deleted_marks_tenant_cancelled(self, client, mock_stripe):
        key = "test_key_sub_cancel_test_000000000"
        customer_id = "cus_sub_cancel"
        _seed_tenant(customer_id, key)

        event = _make_event("customer.subscription.deleted", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        assert webhooks_module._tenant_store[customer_id]["status"] == "cancelled"

    def test_deactivated_key_cannot_authenticate(self, client, mock_stripe):
        key = "test_key_sub_noauth_test_000000000"
        customer_id = "cus_sub_noauth"
        _seed_tenant(customer_id, key)

        event = _make_event("customer.subscription.deleted", {"customer": customer_id})
        mock_stripe.construct_event.return_value = event

        client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})

        resp = client.get("/keys/list", headers={"Authorization": f"Bearer {key}"})
        assert resp.status_code == 403

    def test_subscription_deleted_unknown_customer_is_noop(self, client, mock_stripe):
        event = _make_event("customer.subscription.deleted", {"customer": "cus_unknown_abc"})
        mock_stripe.construct_event.return_value = event

        resp = client.post("/webhooks/stripe", content=b'{}', headers={"stripe-signature": "t=1,v1=x"})
        assert resp.status_code == 200
