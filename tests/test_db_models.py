"""
Tests for SQLAlchemy ORM models using an in-memory SQLite database.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.db.base import Base
from api.db.models import APIKey, Tenant


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.rollback()
    s.close()


def test_create_tenant(session):
    tenant = Tenant(
        stripe_customer_id="cus_test_001",
        email="test@example.com",
        plan="starter",
        status="active",
        extractions_limit=1000,
    )
    session.add(tenant)
    session.commit()

    result = session.query(Tenant).filter_by(stripe_customer_id="cus_test_001").first()
    assert result is not None
    assert result.email == "test@example.com"
    assert result.plan == "starter"
    assert result.status == "active"
    assert result.extractions_limit == 1000
    assert result.extractions_used == 0
    assert result.id is not None


def test_create_api_key_linked_to_tenant(session):
    tenant = Tenant(
        stripe_customer_id="cus_test_002",
        email="apikey@example.com",
        plan="growth",
        status="active",
        extractions_limit=5000,
    )
    session.add(tenant)
    session.flush()

    api_key = APIKey(
        tenant_id=tenant.id,
        key_hash="a" * 64,
        last4="abcd",
        status="active",
    )
    session.add(api_key)
    session.commit()

    result = session.query(APIKey).filter_by(tenant_id=tenant.id).first()
    assert result is not None
    assert result.last4 == "abcd"
    assert result.status == "active"
    assert result.key_hash == "a" * 64
    assert result.tenant_id == tenant.id


def test_tenant_api_key_relationship(session):
    tenant = Tenant(
        stripe_customer_id="cus_test_003",
        email="rel@example.com",
        plan="trial",
        status="active",
        extractions_limit=100,
    )
    session.add(tenant)
    session.flush()

    key1 = APIKey(tenant_id=tenant.id, key_hash="b" * 64, last4="key1", status="active")
    key2 = APIKey(tenant_id=tenant.id, key_hash="c" * 64, last4="key2", status="revoked")
    session.add_all([key1, key2])
    session.commit()

    loaded = session.query(Tenant).filter_by(id=tenant.id).first()
    assert len(loaded.api_keys) == 2
    statuses = {k.status for k in loaded.api_keys}
    assert statuses == {"active", "revoked"}
