"""
ID Card Camera Capture Test
Captures both front and back of an ID card using the camera
"""
from openid import OpenIDClient
from openid.flows.id_card import capture_id_card

# Initialize client with production API
client = OpenIDClient(api_key="your_key", timeout=120)

# Capture ID card (front + back)
# doc_type options: "emirates_id", "driving_license", "auto"
result = capture_id_card(client, doc_type="emirates_id")

if result:
    print("\n" + "="*60)
    print("ID Card extraction completed successfully!")
    print("="*60)
else:
    print("\n" + "="*60)
    print("ID Card capture was cancelled or failed.")
    print("="*60)
