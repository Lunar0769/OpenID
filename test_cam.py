from openid import OpenIDClient
from openid.flows.passport import capture_passport

client = OpenIDClient(api_key="your_key", timeout=120)
result = capture_passport(client)
