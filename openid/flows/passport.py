import sys
import json
from typing import Any
from openid.capture.camera import capture_image
from openid.exceptions import APIConnectionError, APIResponseError
from openid.capture.strict_validation import validate_capture_strict

# Ensure UTF-8 output on Windows terminals (reconfigure is valid at runtime but missing from typeshed)
if hasattr(sys.stdout, "reconfigure"):
    cast_stdout: Any = sys.stdout
    cast_stdout.reconfigure(encoding="utf-8", errors="replace")


def capture_passport(client) -> dict | None:
    """
    Guide the user through capturing a passport image,
    then submit to the API and return the result.

    Args:
        client: An OpenIDClient instance.

    Returns:
        Parsed API response dict, or None if capture was cancelled.
    """
    print("\n-- Passport Capture -----------------------------------------")
    print("  Position the passport inside the guide box.")
    print("  The camera will auto-capture when the image is stable.\n")

    frame, full_frame, filepath = capture_image(label="Passport")

    if frame is None or filepath is None:
        print("  Capture cancelled.")
        return None

    print(f"  Saved: {filepath}")

    # Strict client-side validation — run on the FULL frame (not the guide-box crop)
    # so the crop-check and YOLO detection have correct context.
    print("\n  Validating capture quality...")
    validation_error = validate_capture_strict(full_frame, doc_type="passport")
    if validation_error:
        print("\nValidation Failed:")
        print(json.dumps(validation_error, indent=2, ensure_ascii=False))
        return validation_error

    # API call
    print("\n  Sending to API...")
    try:
        result = client.extract_passport(filepath)
        print("\nExtraction Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    except APIResponseError as e:
        print(f"\nAPI error {e.status_code}: {e.message}")
        if e.request_id:
            print(f"   Request ID: {e.request_id}")
        return None

    except APIConnectionError as e:
        print(f"\nConnection error: {e}")
        return None
