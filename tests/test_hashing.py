"""Unit tests for api.auth.hashing."""

import hashlib

import pytest

from api.auth.hashing import hash_api_key, verify_api_key


def test_hash_returns_64_char_hex():
    result = hash_api_key("test_key_abc123")
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_hash_is_sha256():
    key = "test_key_testkey"
    expected = hashlib.sha256(key.encode()).hexdigest()
    assert hash_api_key(key) == expected


def test_hash_is_deterministic():
    key = "test_key_xyz"
    assert hash_api_key(key) == hash_api_key(key)


def test_verify_correct_key():
    key = "test_key_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6"
    stored = hash_api_key(key)
    assert verify_api_key(key, stored) is True


def test_verify_wrong_key():
    stored = hash_api_key("test_key_correct")
    assert verify_api_key("test_key_wrong", stored) is False


def test_verify_empty_key_does_not_match():
    stored = hash_api_key("test_key_something")
    assert verify_api_key("", stored) is False


def test_init_exports():
    from api.auth import hash_api_key as h, verify_api_key as v, generate_api_key as g
    assert callable(h) and callable(v) and callable(g)
