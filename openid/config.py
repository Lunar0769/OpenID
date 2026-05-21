import os

# Default API base URL — override via env var or OpenIDClient(base_url=...)
DEFAULT_BASE_URL = os.environ.get("OPENID_BASE_URL", "https://jumeirah-ai.testyourapp.online")

# API key — override via env var or OpenIDClient(api_key=...)
DEFAULT_API_KEY = os.environ.get("OPENID_API_KEY", "")

# Default HTTP timeout in seconds
DEFAULT_TIMEOUT = 60
