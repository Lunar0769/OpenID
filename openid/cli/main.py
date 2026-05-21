import sys
import json
import os
import argparse
from typing import Optional

from openid.client import OpenIDClient
from openid.config import DEFAULT_BASE_URL
from openid.exceptions import APIConnectionError, APIResponseError, APIRetryExhaustedError


def _get_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenIDClient:
    """Build an OpenIDClient from args or environment variables."""
    key = api_key or os.environ.get("OPENID_API_KEY", "")
    url = base_url or os.environ.get("OPENID_BASE_URL", DEFAULT_BASE_URL)

    if not key:
        print(
            "Error: API key not configured. Set OPENID_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    return OpenIDClient(api_key=key, base_url=url)


def _print_result(result: dict):
    """Pretty-print the API response."""
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _handle_error(e: Exception):
    """Print a clean error and exit with code 1."""
    if isinstance(e, APIResponseError):
        print(f"Error: {e.message} (HTTP {e.status_code}, code: {e.error_code})", file=sys.stderr)
        if e.request_id:
            print(f"Request ID: {e.request_id}", file=sys.stderr)
    elif isinstance(e, APIRetryExhaustedError):
        print(f"Error: Request failed after {e.attempts} attempts. {e.last_error.message if e.last_error else 'Unknown error'}", file=sys.stderr)
    elif isinstance(e, APIConnectionError):
        print(f"Error: Could not connect to API. {e}", file=sys.stderr)
    else:
        print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


def cmd_capture(args):
    """Handle: openid capture <passport|id> [--file PATH] [--api-key KEY] [--url URL]"""
    if args.doc_type not in ("passport", "id"):
        print("Error: Invalid document type. Use 'passport' or 'id'.", file=sys.stderr)
        sys.exit(1)

    client = _get_client(api_key=getattr(args, "api_key", None), base_url=getattr(args, "url", None))

    if args.doc_type == "passport":
        if args.file:
            try:
                result = client.extract_passport(args.file)
                _print_result(result)
            except Exception as e:
                _handle_error(e)
        else:
            # Camera capture — post-MVP, but keep the hook
            try:
                from openid.flows.passport import capture_passport
                capture_passport(client)
            except ImportError:
                print("Error: Camera capture requires opencv-python. Install it or use --file.", file=sys.stderr)
                sys.exit(1)

    elif args.doc_type == "id":
        if args.file:
            try:
                result = client.extract_id(args.file)
                _print_result(result)
            except Exception as e:
                _handle_error(e)
        else:
            try:
                from openid.flows.id_card import capture_id_card
                capture_id_card(client)
            except ImportError:
                print("Error: Camera capture requires opencv-python. Install it or use --file.", file=sys.stderr)
                sys.exit(1)


def cmd_usage(args):
    """Handle: openid usage [--api-key KEY] [--url URL]"""
    client = _get_client(api_key=getattr(args, "api_key", None), base_url=getattr(args, "url", None))
    try:
        result = client.get_usage()
        _print_result(result)
    except Exception as e:
        _handle_error(e)


def main():
    parser = argparse.ArgumentParser(
        prog="openid",
        description="OpenID Verify — OCR for passports and ID cards",
    )
    parser.add_argument("--api-key", help="API key (overrides OPENID_API_KEY env var)")
    parser.add_argument("--url", help=f"API base URL (default: {DEFAULT_BASE_URL})")

    subparsers = parser.add_subparsers(dest="command")

    # openid capture <passport|id> [--file PATH]
    capture_parser = subparsers.add_parser("capture", help="Capture and extract a document")
    capture_parser.add_argument("doc_type", choices=["passport", "id"], help="Document type")
    capture_parser.add_argument("--file", help="Path to image file (skips camera capture)")

    # openid usage
    subparsers.add_parser("usage", help="Show current API usage and quota")

    args = parser.parse_args()

    if args.command == "capture":
        cmd_capture(args)
    elif args.command == "usage":
        cmd_usage(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
