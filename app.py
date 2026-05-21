import streamlit as st
import json

from openid import OpenIDClient
from openid.flows.passport import capture_passport
from openid.flows.id_card import capture_id_card


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OpenID Test Suite",
    page_icon="🪪",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🪪 OpenID Camera Capture Test Suite")
st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="Enter OpenID API Key"
)

timeout = st.sidebar.number_input(
    "Timeout (seconds)",
    min_value=30,
    max_value=300,
    value=120
)


# ============================================================
# DOCUMENT TYPE
# ============================================================

doc_option = st.selectbox(
    "Select Document Type",
    [
        "Passport",
        "Emirates ID",
        "Driving License",
        "ID Card (Auto-detect)"
    ]
)

# ============================================================
# START BUTTON
# ============================================================

start_btn = st.button("🚀 Start Capture")


# ============================================================
# MAIN LOGIC
# ============================================================

if start_btn:

    if not api_key:
        st.error("Please enter API Key")
        st.stop()

    try:
        # Initialize Client
        client = OpenIDClient(
            api_key=api_key,
            timeout=timeout
        )

        with st.spinner("Starting capture flow..."):

            # ====================================================
            # PASSPORT
            # ====================================================

            if doc_option == "Passport":

                result = capture_passport(client)

            # ====================================================
            # EMIRATES ID
            # ====================================================

            elif doc_option == "Emirates ID":

                result = capture_id_card(
                    client,
                    doc_type="emirates_id"
                )

            # ====================================================
            # DRIVING LICENSE
            # ====================================================

            elif doc_option == "Driving License":

                result = capture_id_card(
                    client,
                    doc_type="driving_license"
                )

            # ====================================================
            # AUTO DETECT
            # ====================================================

            elif doc_option == "ID Card (Auto-detect)":

                result = capture_id_card(
                    client,
                    doc_type="auto"
                )

            else:
                result = None

        # ========================================================
        # RESULT DISPLAY
        # ========================================================

        st.markdown("---")

        if result:

            st.success("✅ Extraction Completed Successfully")

            # RAW JSON
            with st.expander("📦 Full Response JSON", expanded=False):
                st.json(result)

            # OCR DATA
            data = result.get("data", {})
            ocr_data = data.get("ocrData", {})

            if ocr_data:

                st.subheader("📋 Extracted Fields")

                for key, value in ocr_data.items():

                    if (
                        value
                        and key not in [
                            "notExtracted",
                            "ocrDataConfidence"
                        ]
                    ):
                        st.write(f"**{key}:** {value}")

            # CONFIDENCE
            confidence = ocr_data.get("ocrDataConfidence")

            if confidence:
                st.subheader("📊 OCR Confidence")
                st.json(confidence)

        else:
            st.error("❌ Capture failed or cancelled")

    except Exception as e:
        st.exception(e)