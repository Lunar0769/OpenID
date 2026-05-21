from __future__ import annotations
from typing import Optional


class OpenIDError(Exception):
    """Base exception for all OpenID SDK errors."""
    pass


class APIConnectionError(OpenIDError):
    """Raised when a network error or timeout occurs."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class APIResponseError(OpenIDError):
    """Raised when the API returns an HTTP 4xx or 5xx response."""
    def __init__(self, status_code: int, error_code: str, message: str, request_id: Optional[str] = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id


class APIRetryExhaustedError(OpenIDError):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_error: Optional[APIResponseError] = None):
        super().__init__(f"All {attempts} retry attempts failed. Last error: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


class CameraError(OpenIDError):
    """Raised when the webcam is unavailable."""
    pass


class ImageQualityError(OpenIDError):
    """
    Raised when an image fails quality checks and must not be sent to the API.

    Attributes:
        message:  Human-readable description of the failure.
        reason:   Machine-readable reason code, e.g. 'too_blurry', 'low_light', 'no_document'.
        metrics:  Full quality metrics dict (blur_score, brightness, document_detected, document_area_ratio).
    """
    def __init__(self, message: str, reason: str, metrics: dict):
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.metrics = metrics
