"""
OpenID Verify — Streamlit Web UI
=================================
Two capture modes:
  • Upload  — user picks a file from disk
  • Camera  — browser webcam via st.camera_input

Features:
  • Contour + stats overlay drawn on every captured/uploaded image
  • Fullscreen camera mode on mobile via CSS injection
  • Raw OCR section in results (below confidence)
  • API base URL from openid/config.py — no manual entry
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
from openid.capture.quality import check_quality, detect_document_contour
from openid.capture.overlay import get_guide_box
from openid.capture.strict_validation import validate_capture_strict


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenID Verify",
    page_icon="🪪",
    layout="wide",
)

# ── Fullscreen camera CSS ─────────────────────────────────────────────────────
# Makes st.camera_input fill the full viewport on mobile devices.
# On desktop it stays at a comfortable max-width.
st.markdown("""
<style>
/* ── Fullscreen camera on mobile ── */
@media (max-width: 768px) {
    /* Stretch the video preview to full screen width */
    [data-testid="stCameraInput"] video {
        width:  100vw !important;
        height: 56vw !important;   /* 16:9 ratio */
        max-height: 80vh !important;
        object-fit: cover !important;
        border-radius: 0 !important;
    }
    /* Stretch the captured snapshot too */
    [data-testid="stCameraInput"] img {
        width:  100vw !important;
        height: auto !important;
    }
    /* Remove side padding so camera bleeds edge-to-edge */
    section[data-testid="stMain"] > div {
        padding-left:  0 !important;
        padding-right: 0 !important;
    }
}

/* ── Desktop: comfortable max-width ── */
@media (min-width: 769px) {
    [data-testid="stCameraInput"] video {
        width:  100% !important;
        max-width: 720px !important;
        height: auto !important;
        border-radius: 8px;
    }
}

/* ── Overlay image: always full container width ── */
.overlay-img img {
    width: 100% !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

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


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_upload, tab_camera = st.tabs(["📁 Upload File", "📷 Camera"])


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY HELPER — draw guide box + document contour + stats on a BGR frame
# ══════════════════════════════════════════════════════════════════════════════

def _draw_overlay(bgr: np.ndarray, camera_mode: bool = False) -> tuple[np.ndarray, dict]:
    """
    Draw on a copy of bgr:
      • Dimmed region outside the guide box
      • Coloured corner-bracket guide box  (green=pass, yellow=warn, red=fail)
      • Document contour (cyan) if detected
      • Stats panel: Blur / Brightness / Doc / Status
    Returns the annotated BGR image.
    """
    frame = bgr.copy()
    h, w  = frame.shape[:2]

    # ── Quality check ─────────────────────────────────────────────────────────
    q       = check_quality(frame, camera_mode=camera_mode)
    status  = q["status"]
    metrics = q["metrics"]
    reasons = q["reasons"]

    COLOR = {
        "pass":    (0, 220, 80),    # green
        "warning": (0, 200, 255),   # yellow
        "fail":    (0, 60, 255),    # red
    }
    color = COLOR.get(status, COLOR["fail"])

    # ── Guide box ─────────────────────────────────────────────────────────────
    gx, gy, gw, gh = get_guide_box(frame)

    # Dim outside
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[gy:gy+gh, gx:gx+gw] = 255
    dark = (frame * 0.35).astype(np.uint8)
    frame[mask == 0] = dark[mask == 0]

    # Corner brackets
    cl = min(gw, gh) // 6
    tk = max(2, w // 200)
    for (ax, ay), (bx, by), (cx, cy) in [
        ((gx,      gy+cl),  (gx,      gy),      (gx+cl,   gy     )),
        ((gx+gw-cl,gy),     (gx+gw,   gy),      (gx+gw,   gy+cl  )),
        ((gx+gw,   gy+gh-cl),(gx+gw,  gy+gh),   (gx+gw-cl,gy+gh  )),
        ((gx+cl,   gy+gh),  (gx,      gy+gh),   (gx,      gy+gh-cl)),
    ]:
        cv2.line(frame, (ax, ay), (bx, by), color, tk, cv2.LINE_AA)
        cv2.line(frame, (bx, by), (cx, cy), color, tk, cv2.LINE_AA)

    # Thin full outline
    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), color, 1)

    # ── Document contour (cyan) ───────────────────────────────────────────────
    contour, _ = detect_document_contour(frame, camera_mode=camera_mode)
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (200, 200, 0), max(2, tk), cv2.LINE_AA)

    # ── Stats panel (top-left) ────────────────────────────────────────────────
    font      = cv2.FONT_HERSHEY_SIMPLEX
    blur_v    = metrics.get("blur_score", 0)
    bright_v  = metrics.get("brightness", 0)
    doc_v     = metrics.get("document_detected", False)
    doc_pct   = metrics.get("document_area_ratio", 0) * 100

    status_label = {
        "pass":    "PASS",
        "warning": "WARN",
        "fail":    "FAIL",
    }.get(status, "FAIL")
    reason_str = " | ".join(r.replace("_", " ").upper() for r in reasons) if reasons else "OK"

    panel_lines = [
        (f"Status  {status_label}",  color),
        (f"Blur    {blur_v:.0f}",    (255, 255, 255)),
        (f"Light   {bright_v:.0f}",  (255, 255, 255)),
        (f"Doc     {'YES' if doc_v else 'NO'} ({doc_pct:.0f}%)", (255, 255, 255)),
        (reason_str,                 color),
    ]

    scale  = max(0.35, w / 1600)
    lh     = int(22 * scale * (w / 400))
    lh     = max(16, min(lh, 28))
    margin = max(6, w // 80)
    bg_w   = int(w * 0.42)
    bg_h   = len(panel_lines) * lh + margin * 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + bg_w, margin + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (text, tcol) in enumerate(panel_lines):
        y = margin + (i + 1) * lh
        cv2.putText(frame, text, (margin + 4 + 1, y + 1), font, scale * 0.9, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (margin + 4,     y),     font, scale * 0.9, tcol,    1, cv2.LINE_AA)

    return frame, q


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def _display_result(result: dict) -> None:
    """Render extraction result: fields → confidence → raw OCR."""
    st.success("✅ Extraction completed")

    with st.expander("📦 Full JSON response", expanded=False):
        st.json(result)

    # ── Structured fields ─────────────────────────────────────────────────────
    flat_fields  = ["name", "document_number", "dob", "expiry", "country", "document_subtype"]
    flat_data    = {k: v for k, v in result.items() if k in flat_fields and v}
    ocr_data     = result.get("data", {}).get("ocrData", {})
    nested_data  = {
        k: v for k, v in ocr_data.items()
        if v and k not in ("notExtracted", "ocrDataConfidence")
    }
    display_data = flat_data or nested_data

    if display_data:
        st.subheader("📋 Extracted Fields")
        for key, value in display_data.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence = result.get("confidence") or ocr_data.get("ocrDataConfidence")
    if confidence:
        st.subheader("📊 Confidence")
        st.json(confidence)

    # ── Raw OCR ───────────────────────────────────────────────────────────────
    # Collect every raw text field the API returned that wasn't in the
    # structured section — MRZ lines, raw_text, mrz_line1/2, etc.
    raw_candidates = {
        "mrz_line1":    result.get("mrz_line1") or result.get("data", {}).get("mrz_line1"),
        "mrz_line2":    result.get("mrz_line2") or result.get("data", {}).get("mrz_line2"),
        "mrz_line3":    result.get("mrz_line3") or result.get("data", {}).get("mrz_line3"),
        "raw_text":     result.get("raw_text")  or result.get("data", {}).get("raw_text"),
        "raw_mrz":      result.get("raw_mrz")   or result.get("data", {}).get("raw_mrz"),
        "ocr_raw":      result.get("ocr_raw")   or result.get("data", {}).get("ocr_raw"),
        "full_text":    result.get("full_text")  or result.get("data", {}).get("full_text"),
    }
    # Also grab anything in data.ocrData.notExtracted
    not_extracted = ocr_data.get("notExtracted")
    if not_extracted:
        raw_candidates["notExtracted"] = not_extracted

    raw_data = {k: v for k, v in raw_candidates.items() if v}

    # Fallback: if the API returns a flat "raw" key or any key ending in _raw
    for k, v in result.items():
        if k not in flat_fields and k not in ("data", "confidence") and isinstance(v, str) and len(v) > 20:
            raw_data.setdefault(k, v)

    if raw_data:
        st.subheader("🔤 Raw OCR Output")
        for key, value in raw_data.items():
            label = key.replace("_", " ").title()
            if isinstance(value, str):
                st.text_area(label, value=value, height=80, disabled=True, key=f"raw_{key}")
            else:
                st.json({key: value})


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _save_cv2_to_tmp(img_bgr: np.ndarray, suffix: str = ".jpg") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    cv2.imwrite(tmp.name, img_bgr)
    tmp.close()
    return tmp.name


def _save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name


def _show_overlay(bgr: np.ndarray, caption: str = "", camera_mode: bool = False) -> None:
    """Draw overlay on bgr, display in Streamlit, and show quality metrics."""
    annotated, quality = _draw_overlay(bgr, camera_mode=camera_mode)
    # Convert BGR → RGB for st.image
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption, use_container_width=True)

    # ── Quality metrics (matches CLI output) ──────────────────────────────
    status  = quality["status"]
    metrics = quality["metrics"]
    reasons = quality["reasons"]

    cols = st.columns(4)
    status_emoji = {"pass": "🟢", "warning": "🟡", "fail": "🔴"}.get(status, "🔴")
    cols[0].metric("Status", f"{status_emoji} {status.upper()}")
    cols[1].metric("Blur Score", f"{metrics['blur_score']:.0f}")
    cols[2].metric("Brightness", f"{metrics['brightness']:.0f}")
    doc_pct = metrics["document_area_ratio"] * 100
    cols[3].metric("Document", f"{'✓' if metrics['document_detected'] else '✗'} ({doc_pct:.0f}%)")

    if status == "fail":
        reason_str = " · ".join(r.replace("_", " ").title() for r in reasons)
        st.error(f"🚫 Quality FAIL — {reason_str}")
    elif status == "warning":
        reason_str = " · ".join(r.replace("_", " ").title() for r in reasons)
        st.warning(f"⚠️ Quality WARNING — {reason_str}")


def _validate_inputs(key: str) -> bool:
    if not key:
        st.error("Please enter your API key in the sidebar.")
        return False
    return True


def _run_extraction(client: OpenIDClient, tmp_paths: list) -> dict | None:
    if is_id_card:
        if len(tmp_paths) < 2:
            st.error("Need both front and back images.")
            return None
        return client.extract_id(tmp_paths[0], tmp_paths[1], doc_type=DOC_TYPE_MAP[doc_option])
    return client.extract_passport(tmp_paths[0])


def _run_strict_validation(bgr: np.ndarray, doc_type: str = "passport") -> dict | None:
    """
    Run strict post-capture validation (glare, shadow, finger, MRZ).
    Mirrors the CLI's validate_capture_strict() call.
    Returns error dict if validation fails, None if passed.
    """
    validation_error = validate_capture_strict(bgr, doc_type=doc_type)
    if validation_error:
        error_code = validation_error.get("error", "UNKNOWN")
        action = validation_error.get("action", "Please try again.")

        error_labels = {
            "DOCUMENT_NOT_FOUND":    "📄 Document Not Found",
            "DOCUMENT_INVALID":      "📄 Invalid Document",
            "LIGHT_GLARE_DETECTED":  "💡 Glare Detected",
            "SHADOW_DETECTED":       "🌑 Shadow Detected",
            "FINGERS_DETECTED":      "✋ Fingers Detected",
            "MRZ_NOT_VISIBLE":       "🔤 MRZ Not Visible",
        }

        label = error_labels.get(error_code, f"❌ {error_code}")
        st.error(f"{label}: {action}")
        with st.expander("🔍 Validation details", expanded=False):
            st.json(validation_error)
        return validation_error
    st.success("✅ Strict validation passed (glare, shadow, finger, MRZ checks OK)")
    return None


def _handle_errors(exc: Exception) -> None:
    if isinstance(exc, ImageQualityError):
        st.error(f"🚫 Image quality check failed: {exc.message}")
        st.caption(f"Reason: `{exc.reason}` — try better lighting or hold the document flatter.")
    elif isinstance(exc, APIResponseError):
        st.error(f"❌ API error {exc.status_code}: {exc.message}")
        if exc.request_id:
            st.caption(f"Request ID: `{exc.request_id}`")
    elif isinstance(exc, APIConnectionError):
        st.error(f"🔌 Connection error: {exc}")
    else:
        st.exception(exc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

with tab_upload:

    if is_id_card:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Front of ID card**")
            front_file = st.file_uploader("JPEG / PNG, max 10 MB", type=["jpg","jpeg","png"], key="upload_front")
            if front_file:
                _show_overlay(_pil_to_cv2(Image.open(front_file)), "Front — with overlay")
        with col2:
            st.markdown("**Back of ID card**")
            back_file = st.file_uploader("JPEG / PNG, max 10 MB", type=["jpg","jpeg","png"], key="upload_back")
            if back_file:
                _show_overlay(_pil_to_cv2(Image.open(back_file)), "Back — with overlay")
    else:
        passport_file = st.file_uploader("Passport image (JPEG / PNG, max 10 MB)", type=["jpg","jpeg","png"], key="upload_passport")
        if passport_file:
            _show_overlay(_pil_to_cv2(Image.open(passport_file)), "Passport — with overlay")

    st.markdown("---")
    upload_btn = st.button("🔍 Extract from Upload", type="primary", key="btn_upload")

    if upload_btn:
        if not _validate_inputs(api_key):
            st.stop()
        if is_id_card and (not front_file or not back_file):
            st.error("Please upload both front and back images.")
            st.stop()
        if not is_id_card and not passport_file:
            st.error("Please upload a passport image.")
            st.stop()

        # ── Strict validation (same checks as CLI) ───────────────────
        st.subheader("🔬 Strict Validation")
        doc_type_val = DOC_TYPE_MAP.get(doc_option, "passport") if is_id_card else "passport"

        if is_id_card:
            st.caption("Front side:")
            front_file.seek(0)
            if _run_strict_validation(_pil_to_cv2(Image.open(front_file)), doc_type=doc_type_val):
                st.stop()
            st.caption("Back side:")
            back_file.seek(0)
            if _run_strict_validation(_pil_to_cv2(Image.open(back_file)), doc_type=doc_type_val):
                st.stop()
        else:
            passport_file.seek(0)
            if _run_strict_validation(_pil_to_cv2(Image.open(passport_file)), doc_type="passport"):
                st.stop()

        # ── Extract via API ──────────────────────────────────────────
        client = OpenIDClient(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=timeout)
        tmp_paths = []
        try:
            with st.spinner("Extracting..."):
                tmp_paths = ([_save_upload(front_file), _save_upload(back_file)]
                             if is_id_card else [_save_upload(passport_file)])
                result = _run_extraction(client, tmp_paths)
            if result:
                _display_result(result)
        except Exception as e:
            _handle_errors(e)
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except OSError: pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAMERA
# ══════════════════════════════════════════════════════════════════════════════

with tab_camera:

    st.info(
        "📷 Uses your **browser camera** — works on phone and desktop. "
        "The preview fills the screen on mobile. "
        "Point at the document and tap the capture button."
    )

    if is_id_card:
        st.markdown("### Step 1 — Front of ID card")
        front_cam = st.camera_input("Capture front", key="cam_front")
        if front_cam:
            front_bgr = _pil_to_cv2(Image.open(front_cam))
            _show_overlay(front_bgr, "Front — quality overlay", camera_mode=True)

        st.markdown("### Step 2 — Back of ID card")
        back_cam = st.camera_input("Capture back", key="cam_back")
        if back_cam:
            back_bgr = _pil_to_cv2(Image.open(back_cam))
            _show_overlay(back_bgr, "Back — quality overlay", camera_mode=True)

    else:
        passport_cam = st.camera_input("Capture passport", key="cam_passport")
        if passport_cam:
            passport_bgr = _pil_to_cv2(Image.open(passport_cam))
            _show_overlay(passport_bgr, "Passport — quality overlay", camera_mode=True)

    st.markdown("---")
    camera_btn = st.button("🔍 Extract from Camera", type="primary", key="btn_camera")

    if camera_btn:
        if not _validate_inputs(api_key):
            st.stop()
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

        # ── Strict validation (same checks as CLI camera flow) ────
        st.subheader("🔬 Strict Validation")
        doc_type_val = DOC_TYPE_MAP.get(doc_option, "passport") if is_id_card else "passport"

        if is_id_card:
            st.caption("Front side:")
            front_cam.seek(0)
            if _run_strict_validation(_pil_to_cv2(Image.open(front_cam)), doc_type=doc_type_val):
                st.stop()
            st.caption("Back side:")
            back_cam.seek(0)
            if _run_strict_validation(_pil_to_cv2(Image.open(back_cam)), doc_type=doc_type_val):
                st.stop()
        else:
            passport_cam.seek(0)
            if _run_strict_validation(_pil_to_cv2(Image.open(passport_cam)), doc_type="passport"):
                st.stop()

        # ── Extract via API ──────────────────────────────────────────
        client = OpenIDClient(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=timeout)
        tmp_paths = []
        try:
            with st.spinner("Extracting..."):
                if is_id_card:
                    front_cam.seek(0)
                    back_cam.seek(0)
                    tmp_paths = [
                        _save_cv2_to_tmp(_pil_to_cv2(Image.open(front_cam))),
                        _save_cv2_to_tmp(_pil_to_cv2(Image.open(back_cam))),
                    ]
                else:
                    passport_cam.seek(0)
                    tmp_paths = [_save_cv2_to_tmp(_pil_to_cv2(Image.open(passport_cam)))]
                result = _run_extraction(client, tmp_paths)
            if result:
                _display_result(result)
        except Exception as e:
            _handle_errors(e)
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except OSError: pass
