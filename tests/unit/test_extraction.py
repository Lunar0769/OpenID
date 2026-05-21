"""
Unit tests for extraction endpoints.

Tests the API layer with a mocked OCR pipeline.
"""

import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api.auth.middleware import _key_store
from api.auth.hashing import hash_api_key


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_key_store():
    _key_store.clear()
    yield
    _key_store.clear()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def auth_key():
    key = "test_key_testextract0000000000000000"
    _key_store[hash_api_key(key)] = {
        "tenant_id": "t_extract_test",
        "plan": "starter",
        "key_hash": hash_api_key(key),
    }
    return key


def _make_jpeg(size_bytes: int = 1024) -> bytes:
    """Return a minimal valid JPEG-like byte sequence for testing."""
    # JPEG magic bytes + padding
    header = b"\xff\xd8\xff\xe0" + b"\x00" * 16
    padding = b"\x00" * max(0, size_bytes - len(header))
    return header + padding


def _make_upload(data: bytes, filename: str = "test.jpg", content_type: str = "image/jpeg"):
    """Return a files dict suitable for TestClient.post(files=...)."""
    return {"file": (filename, io.BytesIO(data), content_type)}


# ---------------------------------------------------------------------------
# POST /extract — happy path
# ---------------------------------------------------------------------------

class TestExtractHappyPath:
    def test_valid_jpeg_returns_200_with_schema(self, client, auth_key):
        mock_result = {
            "name": "JOHN DOE",
            "document_number": "A1234567",
            "dob": "1990-01-15",
            "expiry": "2030-01-15",
            "country": "USA",
        }
        with patch("app.extraction.pipeline.extract_document", return_value=mock_result), \
             patch("cv2.imdecode", return_value=MagicMock()):
            resp = client.post(
                "/extract",
                files=_make_upload(_make_jpeg()),
                headers={"Authorization": f"Bearer {auth_key}"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body
        assert "document_number" in body
        assert "dob" in body
        assert "expiry" in body
        assert "country" in body

    def test_valid_jpeg_returns_correct_values(self, client, auth_key):
        mock_result = {
            "name": "JANE SMITH",
            "document_number": "B9876543",
            "dob": "1985-06-20",
            "expiry": "2028-06-20",
            "country": "GBR",
        }
        with patch("app.extraction.pipeline.extract_document", return_value=mock_result), \
             patch("cv2.imdecode", return_value=MagicMock()):
            resp = client.post(
                "/extract",
                files=_make_upload(_make_jpeg()),
                headers={"Authorization": f"Bearer {auth_key}"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "JANE SMITH"
        assert body["document_number"] == "B9876543"


# ---------------------------------------------------------------------------
# POST /extract — file too large
# ---------------------------------------------------------------------------

class TestExtractFileTooLarge:
    def test_file_over_10mb_returns_413(self, client, auth_key):
        large_data = b"\xff\xd8\xff\xe0" + b"\x00" * (10 * 1024 * 1024 + 1)
        resp = client.post(
            "/extract",
            files=_make_upload(large_data),
            headers={"Authorization": f"Bearer {auth_key}"},
        )
        assert resp.status_code == 413
        assert resp.json()["detail"]["error"] == "file_too_large"

    def test_file_exactly_10mb_is_allowed(self, client, auth_key):
        # Exactly 10 MB should pass validation (not exceed)
        exact_data = b"\xff\xd8\xff\xe0" + b"\x00" * (10 * 1024 * 1024 - 4)
        mock_result = {"name": "X", "document_number": "Y", "dob": "2000-01-01", "expiry": "2030-01-01", "country": "USA"}
        with patch("app.extraction.pipeline.extract_document", return_value=mock_result), \
             patch("cv2.imdecode", return_value=MagicMock()):
            resp = client.post(
                "/extract",
                files=_make_upload(exact_data),
                headers={"Authorization": f"Bearer {auth_key}"},
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /extract — wrong format
# ---------------------------------------------------------------------------

class TestExtractWrongFormat:
    def test_pdf_returns_422_unsupported_format(self, client, auth_key):
        pdf_data = b"%PDF-1.4 fake pdf content"
        resp = client.post(
            "/extract",
            files=_make_upload(pdf_data, filename="doc.pdf", content_type="application/pdf"),
            headers={"Authorization": f"Bearer {auth_key}"},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "unsupported_format"

    def test_text_file_returns_422_unsupported_format(self, client, auth_key):
        resp = client.post(
            "/extract",
            files=_make_upload(b"hello world", filename="doc.txt", content_type="text/plain"),
            headers={"Authorization": f"Bearer {auth_key}"},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "unsupported_format"


# ---------------------------------------------------------------------------
# POST /extract — no document detected
# ---------------------------------------------------------------------------

class TestExtractNoDocument:
    def test_no_document_detected_returns_422(self, client, auth_key):
        with patch("app.extraction.pipeline.extract_document", side_effect=ValueError("no_document_detected")), \
             patch("cv2.imdecode", return_value=MagicMock()):
            resp = client.post(
                "/extract",
                files=_make_upload(_make_jpeg()),
                headers={"Authorization": f"Bearer {auth_key}"},
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "no_document_detected"

    def test_undecodable_image_returns_422(self, client, auth_key):
        with patch("cv2.imdecode", return_value=None):
            resp = client.post(
                "/extract",
                files=_make_upload(b"\xff\xd8\xff\xe0" + b"\x00" * 100),
                headers={"Authorization": f"Bearer {auth_key}"},
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "no_document_detected"


# ---------------------------------------------------------------------------
# POST /extract — auth required
# ---------------------------------------------------------------------------

class TestExtractAuth:
    def test_extract_requires_auth(self, client):
        resp = client.post("/extract", files=_make_upload(_make_jpeg()))
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /extract-id — missing back image
# ---------------------------------------------------------------------------

class TestExtractIdMissingBack:
    def test_missing_back_image_returns_422(self, client, auth_key):
        with patch("app.extraction.pipeline.extract_document", side_effect=ValueError("missing_back_image")), \
             patch("cv2.imdecode", return_value=MagicMock()):
            resp = client.post(
                "/extract-id",
                files=[("front", ("front.jpg", io.BytesIO(_make_jpeg()), "image/jpeg"))],
                headers={"Authorization": f"Bearer {auth_key}"},
            )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "missing_back_image"

    def test_extract_id_happy_path(self, client, auth_key):
        mock_result = {
            "name": "ALI HASSAN",
            "document_number": "784-1990-1234567-1",
            "dob": "1990-03-10",
            "expiry": "2025-03-10",
            "country": "ARE",
        }
        with patch("app.extraction.pipeline.extract_document", return_value=mock_result), \
             patch("cv2.imdecode", return_value=MagicMock()):
            resp = client.post(
                "/extract-id",
                files=[
                    ("front", ("front.jpg", io.BytesIO(_make_jpeg()), "image/jpeg")),
                    ("back", ("back.jpg", io.BytesIO(_make_jpeg()), "image/jpeg")),
                ],
                headers={"Authorization": f"Bearer {auth_key}"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "ALI HASSAN"

    def test_extract_id_requires_auth(self, client):
        resp = client.post(
            "/extract-id",
            files=[("front", ("front.jpg", io.BytesIO(_make_jpeg()), "image/jpeg"))],
        )
        assert resp.status_code == 401
