import json
from openid.capture.camera import capture_image, wait_for_flip
from openid.exceptions import APIConnectionError, APIResponseError


def capture_id_card(client, doc_type: str = "emirates_id") -> dict | None:
    """
    Guide the user through capturing both sides of an ID card,
    then submit to the API and return the result.

    Args:
        client:   An OpenIDClient instance.
        doc_type: Document type hint passed to the API (default: "emirates_id").

    Returns:
        Parsed API response dict, or None if capture was cancelled.
    """
    print("\n── ID Card Capture ──────────────────────────────────────")
    print("  Position the FRONT of the card in the frame.")
    print("  The camera will auto-capture when the image is clear.\n")

    # ── Front side ────────────────────────────────────────────────────────────
    front_frame, full_front_frame, front_path = capture_image(label="ID Front")

    if front_frame is None or front_path is None:
        print("❌ Front capture cancelled.")
        return None

    print(f"  ✅ Front saved: {front_path}")

    # ── Flip prompt ───────────────────────────────────────────────────────────
    continued = wait_for_flip()
    if not continued:
        print("❌ Capture cancelled during flip.")
        return None

    # ── Back side ─────────────────────────────────────────────────────────────
    print("\n  Position the BACK of the card in the frame.\n")
    back_frame, full_back_frame, back_path = capture_image(label="ID Back")

    if back_frame is None or back_path is None:
        print("❌ Back capture cancelled.")
        return None

    print(f"  ✅ Back saved:  {back_path}")

    # ── API call ──────────────────────────────────────────────────────────────
    print("\n  🔄  Sending to API...")
    try:
        result = client.extract_id(front_path, back_path, doc_type=doc_type)
        print("\n✅ Extraction Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    except APIResponseError as e:
        print(f"\n❌ API error {e.status_code}: {e.message}")
        if e.request_id:
            print(f"   Request ID: {e.request_id}")
        return None

    except APIConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        return None
