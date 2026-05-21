from openid import OpenIDClient
from openid.exceptions import APIConnectionError, APIResponseError
import json

print("Loading OpenID........")

client = OpenIDClient(api_key="test", base_url="https://jumeirah-ai.testyourapp.online", timeout=60)

print("OpenID Loaded.....")

# --- Passport ---
try:

    print("Calling API.......")

    result = client.extract_passport("passport.jpg")

    print("Passport:")
    print(json.dumps(result, indent=4))

except APIConnectionError as e:
    print(f"Server not running: {e}")

except APIResponseError as e:
    print(f"API error {e.status_code}: {e.message}")

# --- ID card via /extract-id (front + back) ---
# try:
#     result = client.extract_id("id_front.jpg", "id_back.jpg", doc_type="emirates_id")
#     print("ID card:", result)
# except APIResponseError as e:
#     print(f"API error {e.status_code}: {e.message}")
