"""
OpenID Verify — Streamlit Web UI
=================================
File-upload mode for hosted/cloud deployment (no webcam required).
"""

import json
import tempfile
import os
import streamlit as st

from openid import OpenIDClient
from openid.exceptions import APIConnectionError, APIResponseError, ImageQualityError


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenID Verify",
    page_icon="🪪",
    layout="wide",
)

st.title("🪪 OpenID Verify — OCR Extraction")
st.markdown("Upload a passport or ID card image to extract structured data.")
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="Enter your OpenID API key",
)

api_url = st.sidebar.text_input(
    "API URL",
    value=os.environ.get("OPENID_API_URL", "https://api.openid.ai"),
)

timeout = st.sidebar.number_input(
    "Timeout (seconds)",
    min_value=10,
    max_value=300,
    value=30,
)

st.sidebar.markdown("---")
st.sidebar.caption("Camera capture is not available in the hosted version. Upload an image file instead.")


# ── Document type ─────────────────────────────────────────────────────────────

doc_option = st.selectbox(
    "Document Type",
    ["Passport", "Emirates ID", "Driving License", "ID Card (Auto-detect)"],
)

is_id_card = doc_option != "Passport"

DOC_TYPE_MAP = {
    "Emirates ID":        "emirates_id",
    "Driving License":    "driving_license",
    "ID Card (Auto-detect)": "auto",
}


# ── File uploaders ────────────────────────────────────────────────────────────

if is_id_card:
    col1, col2 = st.columns(2)
    with col1:
        front_file = st.file_uploader(
            "Front of ID card (JPEG / PNG, max 10 MB)",
            type=["jpg", "jpeg", "png"],
            key="front",
        )
    with col2:
        back_file = st.file_uploader(
            "Back of ID card (JPEG / PNG, max 10 MB)",
            type=["jpg", "jpeg", "png"],
            key="back",
        )
else:
    passport_file = st.file_uploader(
        "Passport image (JPEG / PNG, max 10 MB)",
        type=["jpg", "jpeg", "png"],
        key="passport",
    )


# ── Extract button ────────────────────────────────────────────────────────────

extract_btn = st.button("🔍 Extract", type="primary")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a named temp file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _display_result(result: dict) -> None:
    """Render the extraction result in the UI."""
    st.success("✅ Extraction completed")

    with st.expander("📦 Full JSON response", expanded=False):
        st.json(result)

    # Flat top-level fields (passport response)
    flat_fields = ["name", "document_number", "dob", "expiry", "country", "document_subtype"]
    flat_data = {k: v for k, v in result.items() if k in flat_fields and v}

    # Nested ocrData (some API versions)
    ocr_data = result.get("data", {}).get("ocrData", {})

    display_data = flat_data or {
        k: v for k, v in ocr_data.items()
        if v and k not in ("notExtracted", "ocrDataConfidence")
    }

    if display_data:
        st.subheader("📋 Extracted Fields")
        for key, value in display_data.items():
            label = key.replace("_", " ").title()
            st.write(f"**{label}:** {value}")

    confidence = result.get("confidence") or ocr_data.get("ocrDataConfidence")
    if confidence:
        st.subheader("📊 Confidence")
        st.json(confidence)


# ── Main logic ────────────────────────────────────────────────────────────────

if extract_btn:

    # Validate inputs
    if not api_key:
        st.error("Please enter your API key in the sidebar.")
        st.stop()

    if is_id_card:
        if not front_file:
            st.error("Please upload the front image of the ID card.")
            st.stop()
        if not back_file:
            st.error("Please upload the back image of the ID card.")
            st.stop()
    else:
        if not passport_file:
            st.error("Please upload a passport image.")
            st.stop()

    # Run extraction
    client = OpenIDClient(api_key=api_key, base_url=api_url, timeout=timeout)
    tmp_paths = []

    try:
        with st.spinner("Extracting document data..."):

            if is_id_card:
                front_path = _save_upload(front_file)
                back_path  = _save_upload(back_file)
                tmp_paths  = [front_path, back_path]
                doc_type   = DOC_TYPE_MAP[doc_option]
                result     = client.extract_id(front_path, back_path, doc_type=doc_type)
            else:
                passport_path = _save_upload(passport_file)
                tmp_paths     = [passport_path]
                result        = client.extract_passport(passport_path)

        _display_result(result)

    except ImageQualityError as e:
        st.error(f"🚫 Image quality check failed: {e.message}")
        st.caption(f"Reason: `{e.reason}` — try a clearer, well-lit photo of the document.")

    except APIResponseError as e:
        st.error(f"❌ API error {e.status_code}: {e.message}")
        if e.request_id:
            st.caption(f"Request ID: `{e.request_id}`")

    except APIConnectionError as e:
        st.error(f"🔌 Connection error: {e}")
        st.caption("Check that the API URL is correct and the service is reachable.")

    except Exception as e:
        st.exception(e)

    finally:
        # Clean up temp files
        for path in tmp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
