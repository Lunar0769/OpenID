import sys
import time
import requests
import cv2
from typing import Optional

from openid.config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from openid.exceptions import (
    APIConnectionError,
    APIResponseError,
    APIRetryExhaustedError,
    ImageQualityError,
)

# Status codes that are safe to retry
_RETRYABLE_CODES = {429, 503}


class OpenIDClient:
    """
    Client for the OpenID Verify OCR API.

    Usage:
        from openid import OpenIDClient

        client = OpenIDClient(api_key="sk_live_...")
        result = client.extract_passport("passport.jpg")
        print(result["name"], result["document_number"])
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        if not api_key:
            raise ValueError(
                "api_key is required. Set OPENID_API_KEY env var or pass it directly."
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def extract_passport(self, image_path: str) -> dict:
        """
        Submit a passport image for OCR extraction.

        Quality gate (PRD hard rule — MUST NOT send fail images to API):
            fail    → raises ImageQualityError immediately
            warning → logs to stderr, request still proceeds
            pass    → proceeds silently

        Args:
            image_path: Path to a JPEG or PNG image file (max 10 MB).

        Returns:
            dict with extracted passport fields.

        Raises:
            ImageQualityError:      Image failed quality checks (never reaches API).
            APIConnectionError:     Network failure or timeout.
            APIResponseError:       API returned 4xx/5xx.
            APIRetryExhaustedError: Retryable error persisted after 3 attempts.
        """
        self._quality_gate(image_path)
        with open(image_path, "rb") as f:
            files = {"front_file": (image_path, f, "image/jpeg")}
            return self._post_with_retry("/extract_passport", files=files)

    def extract_id(self, front_path: str, back_path: Optional[str] = None, doc_type: str = "emirates_id") -> dict:
        """
        Submit ID card images for OCR extraction.

        Quality gate applied to both front and back images (PRD hard rule).
        Supported doc_type values: "emirates_id", "driving_license", "auto"

        Args:
            front_path: Path to the front image (JPEG or PNG, max 10 MB).
            back_path:  Path to the back image (JPEG or PNG, max 10 MB).
            doc_type:   Document type hint (default: "emirates_id").

        Returns:
            dict with extracted ID card fields.

        Raises:
            ImageQualityError:      Image failed quality checks (never reaches API).
            APIConnectionError:     Network failure or timeout.
            APIResponseError:       API returned 4xx/5xx.
            APIRetryExhaustedError: Retryable error persisted after 3 attempts.
        """
        self._quality_gate(front_path)
        if back_path:
            self._quality_gate(back_path)
        with open(front_path, "rb") as f:
            files = {"front_file": (front_path, f, "image/jpeg")}
            if back_path:
                with open(back_path, "rb") as b:
                    files["back_file"] = (back_path, b, "image/jpeg")
                    return self._post_with_retry("/extract_id", files=files, params={"doc_type": doc_type})
            return self._post_with_retry("/extract_id", files=files, params={"doc_type": doc_type})

    def get_usage(self) -> dict:
        """
        Query current usage and quota for this API key.

        Returns:
            dict with keys: plan, period, extractions_used, extractions_limit, remaining

        Raises:
            APIConnectionError: Network failure or timeout.
            APIResponseError: API returned 4xx/5xx.
        """
        return self._get("/usage")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quality_gate(self, image_path: str) -> None:
        """
        Run quality check on an image file BEFORE sending it to the API.

        PRD Hard Rule: SDK MUST NOT send images with status='fail'.

        Raises:
            ImageQualityError: If the image fails quality checks.
        """
        from openid.capture.quality import check_quality
        _reason_messages = {
            "blank_frame":     "Image appears blank — ensure the document is visible",
            "too_blurry":     "Image is too blurry — hold the camera steady",
            "slight_blur":    "Image has slight blur",
            "low_light":      "Image has insufficient lighting",
            "dim_light":      "Image lighting is dim",
            "too_bright":     "Image is overexposed — reduce glare or ambient light",
            "no_document":    "No identity document detected in image",
            "person_detected": "A person was detected instead of a document — submit a flat scan of the ID",
        }

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        result  = check_quality(image, camera_mode=False)
        status  = result["status"]
        metrics = result["metrics"]

        if status == "fail":
            reasons        = result["reasons"]
            primary_reason = reasons[0] if reasons else "unknown"
            message        = _reason_messages.get(primary_reason, "Image failed quality checks")
            raise ImageQualityError(message=message, reason=primary_reason, metrics=metrics)

        if status == "warning":
            print(
                f"Warning: image quality suboptimal — {', '.join(result['reasons'])}. "
                f"Metrics: blur={metrics['blur_score']}, brightness={metrics['brightness']}",
                file=sys.stderr,
            )

    def _post_with_retry(self, path: str, files: dict, params: Optional[dict] = None) -> dict:
        """POST with exponential backoff retry for 429/503."""
        max_attempts = 3
        base_delay = 1.0
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self._session.post(
                    f"{self.base_url}{path}",
                    files=files,
                    params=params,
                    timeout=self.timeout,
                )
            except requests.exceptions.RequestException as e:
                raise APIConnectionError(str(e), e)

            if response.status_code in _RETRYABLE_CODES:
                last_error = self._parse_error(response)
                if attempt < max_attempts - 1:
                    retry_after = float(
                        response.headers.get("Retry-After", base_delay * (2 ** attempt))
                    )
                    time.sleep(retry_after)
                    continue
                raise APIRetryExhaustedError(max_attempts, last_error)

            if response.status_code >= 400:
                raise self._parse_error(response)

            return response.json()

        raise APIRetryExhaustedError(max_attempts, last_error)

    def _get(self, path: str) -> dict:
        """Simple GET request, no retry."""
        try:
            response = self._session.get(
                f"{self.base_url}{path}",
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise APIConnectionError(str(e), e)

        if response.status_code >= 400:
            raise self._parse_error(response)

        return response.json()

    def _parse_error(self, response: requests.Response) -> APIResponseError:
        """Parse an error response into an APIResponseError."""
        try:
            body = response.json()
        except Exception:
            body = {}

        return APIResponseError(
            status_code=response.status_code,
            error_code=body.get("error", "unknown_error"),
            message=body.get("message", response.text),
            request_id=response.headers.get("X-Request-ID"),
        )
