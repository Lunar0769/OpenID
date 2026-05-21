"""
OpenID Verify — Streamlit Web UI
=================================
Two capture modes:
  • Upload  — user picks a file from disk
  • Camera  — browser webcam via st.camera_input (works on Render, no server camera needed)

API base URL comes from openid/config.py — no manual entry needed.
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from openid import OpenIDClient
from openid.config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from openid.exceptions import APIConnectionError, APIResponseError, ImageQualityError
from openid.capture.quality import check_quality


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenID Verify",
    page_icon="🪪",
    layout="wide",
)

st.title("🪪 OpenID Verify — OCR Extraction")
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="Enter your OpenID API key",
)

timeout = st.sidebar.number_input(
    "Timeout (seconds)",
    min_value=10,
    max_value=300,
    value=DEFAULT_TIMEOUT,
)

st.sidebar.markdown("---")
st.sidebar.caption(f"API: `{DEFAULT_BASE_URL}`")


# ── Document type ─────────────────────────────────────────────────────────────

doc_option = st.selectbox(
    "Document Type",
    ["Passport", "Emirates ID", "Driving License", "ID Card (Auto-detect)"],
)

is_id_card = doc_option != "Passport"

DOC_TYPE_MAP = {
    "Emirates ID":           "emirates_id",
    "Driving License":       "driving_license",
    "ID Card (Auto-detect)": "auto",
}


# ── Capture mode tabs ─────────────────────────────────────────────────────────

tab_upload, tab_camera = st.tabs(["📁 Upload File", "📷 Camera"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image (RGB) to OpenCV BGR array."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _save_cv2_to_tmp(img_bgr: np.ndarray, suffix: str = ".jpg") -> str:
    """Write a BGR numpy array to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    cv2.imwrite(tmp.name, img_bgr)
    tmp.close()
    return tmp.name


def _save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a named temp file and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _quality_badge(result: dict) -> None:
    """Show a small quality status badge."""
    status  = result["status"]
    reasons = result["reasons"]
    metrics = result["metrics"]

    if status == "pass":
        st.success(f"✅ Quality: PASS  —  blur={metrics['blur_score']}, brightness={metrics['brightness']}")
    elif status == "warning":
        st.warning(f"⚠️ Quality: WARNING  —  {', '.join(reasons)}")
    else:
        st.error(f"🚫 Quality: FAIL  —  {', '.join(reasons)}")


def _display_result(result: dict) -> None:
    """Render the extraction result."""
    st.success("✅ Extraction completed")

    with st.expander("📦 Full JSON response", expanded=False):
        st.json(result)

    flat_fields = ["name", "document_number", "dob", "expiry", "country", "document_subtype"]
    flat_data   = {k: v for k, v in result.items() if k in flat_fields and v}

    ocr_data    = result.get("data", {}).get("ocrData", {})
    nested_data = {
        k: v for k, v in ocr_data.items()
        if v and k not in ("notExtracted", "ocrDataConfidence")
    }

    display_data = flat_data or nested_data

    if display_data:
        st.subheader("📋 Extracted Fields")
        for key, value in display_data.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    confidence = result.get("confidence") or ocr_data.get("ocrDataConfidence")
    if confidence:
        st.subheader("📊 Confidence")
        st.json(confidence)


def _run_extraction(client: OpenIDClient, tmp_paths: list[str]) -> dict | None:
    """Call the right API method based on doc_option and return result."""
    if is_id_card:
        if len(tmp_paths) < 2:
            st.error("Need both front and back images for ID card extraction.")
            return None
        return client.extract_id(
            tmp_paths[0],
            tmp_paths[1],
            doc_type=DOC_TYPE_MAP[doc_option],
        )
    else:
        return client.extract_passport(tmp_paths[0])


def _validate_inputs(api_key: str) -> bool:
    if not api_key:
        st.error("Please enter your API key in the sidebar.")
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:

    if is_id_card:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Front of ID card**")
            front_file = st.file_uploader(
                "JPEG / PNG, max 10 MB",
                type=["jpg", "jpeg", "png"],
                key="upload_front",
            )
            if front_file:
                st.image(front_file, caption="Front", use_container_width=True)
        with col2:
            st.markdown("**Back of ID card**")
            back_file = st.file_uploader(
                "JPEG / PNG, max 10 MB",
                type=["jpg", "jpeg", "png"],
                key="upload_back",
            )
            if back_file:
                st.image(back_file, caption="Back", use_container_width=True)
    else:
        passport_file = st.file_uploader(
            "Passport image (JPEG / PNG, max 10 MB)",
            type=["jpg", "jpeg", "png"],
            key="upload_passport",
        )
        if passport_file:
            st.image(passport_file, caption="Uploaded passport", use_container_width=True)

    st.markdown("---")
    upload_btn = st.button("🔍 Extract from Upload", type="primary", key="btn_upload")

    if upload_btn:
        if not _validate_inputs(api_key):
            st.stop()

        if is_id_card:
            if not front_file or not back_file:
                st.error("Please upload both front and back images.")
                st.stop()
        else:
            if not passport_file:
                st.error("Please upload a passport image.")
                st.stop()

        client    = OpenIDClient(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=timeout)
        tmp_paths = []

        try:
            with st.spinner("Extracting..."):
                if is_id_card:
                    tmp_paths = [_save_upload(front_file), _save_upload(back_file)]
                else:
                    tmp_paths = [_save_upload(passport_file)]

                result = _run_extraction(client, tmp_paths)

            if result:
                _display_result(result)

        except ImageQualityError as e:
            st.error(f"🚫 Image quality check failed: {e.message}")
            st.caption(f"Reason: `{e.reason}`")
        except APIResponseError as e:
            st.error(f"❌ API error {e.status_code}: {e.message}")
            if e.request_id:
                st.caption(f"Request ID: `{e.request_id}`")
        except APIConnectionError as e:
            st.error(f"🔌 Connection error: {e}")
        except Exception as e:
            st.exception(e)
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAMERA (browser webcam via st.camera_input)
# ══════════════════════════════════════════════════════════════════════════════

with tab_camera:

    st.info(
        "📷 Uses your **browser's camera** — works on any device. "
        "Point the camera at the document, wait for a clear shot, then click the capture button."
    )

    if is_id_card:
        st.markdown("### Step 1 — Capture Front")
        front_cam = st.camera_input("Front of ID card", key="cam_front")

        st.markdown("### Step 2 — Capture Back")
        back_cam = st.camera_input("Back of ID card", key="cam_back")

        # Live quality preview for front
        if front_cam:
            front_pil = Image.open(front_cam)
            front_bgr = _pil_to_cv2(front_pil)
            q = check_quality(front_bgr, camera_mode=False)
            _quality_badge(q)

    else:
        passport_cam = st.camera_input("Passport", key="cam_passport")

        # Live quality preview
        if passport_cam:
            pil  = Image.open(passport_cam)
            bgr  = _pil_to_cv2(pil)
            q    = check_quality(bgr, camera_mode=False)
            _quality_badge(q)

    st.markdown("---")
    camera_btn = st.button("🔍 Extract from Camera", type="primary", key="btn_camera")

    if camera_btn:
        if not _validate_inputs(api_key):
            st.stop()

        # Check images are captured
        if is_id_card:
            if not front_cam:
                st.error("Please capture the front of the ID card first.")
                st.stop()
            if not back_cam:
                st.error("Please capture the back of the ID card.")
                st.stop()
        else:
            if not passport_cam:
                st.error("Please capture the passport first.")
                st.stop()

        client    = OpenIDClient(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=timeout)
        tmp_paths = []

        try:
            with st.spinner("Extracting..."):

                if is_id_card:
                    front_bgr = _pil_to_cv2(Image.open(front_cam))
                    back_bgr  = _pil_to_cv2(Image.open(back_cam))
                    tmp_paths = [
                        _save_cv2_to_tmp(front_bgr),
                        _save_cv2_to_tmp(back_bgr),
                    ]
                else:
                    passport_bgr = _pil_to_cv2(Image.open(passport_cam))
                    tmp_paths    = [_save_cv2_to_tmp(passport_bgr)]

                result = _run_extraction(client, tmp_paths)

            if result:
                _display_result(result)

        except ImageQualityError as e:
            st.error(f"🚫 Image quality check failed: {e.message}")
            st.caption(f"Reason: `{e.reason}` — try better lighting or hold the document flatter.")
        except APIResponseError as e:
            st.error(f"❌ API error {e.status_code}: {e.message}")
            if e.request_id:
                st.caption(f"Request ID: `{e.request_id}`")
        except APIConnectionError as e:
            st.error(f"🔌 Connection error: {e}")
        except Exception as e:
            st.exception(e)
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
