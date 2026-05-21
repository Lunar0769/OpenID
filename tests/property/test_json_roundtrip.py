"""
Property-based tests for JSON serialization round-trip.

# Feature: openid-ocr-platform, Property 2: JSON Serialization Round-Trip
"""

import json
import pytest
from pydantic import BaseModel
from hypothesis import given, settings
from hypothesis import strategies as st


class ExtractionResponse(BaseModel):
    name: str
    document_number: str
    dob: str
    expiry: str
    country: str


# Feature: openid-ocr-platform, Property 2: JSON Serialization Round-Trip
@given(
    st.builds(
        ExtractionResponse,
        name=st.text(min_size=1, max_size=100),
        document_number=st.text(min_size=1, max_size=20),
        dob=st.dates().map(str),
        expiry=st.dates().map(str),
        country=st.text(min_size=3, max_size=3),
    )
)
@settings(max_examples=100)
def test_extraction_response_json_roundtrip(response: ExtractionResponse):
    """Validates: Requirements 2.1 — JSON Serialization Round-Trip.

    For any valid ExtractionResponse, parsing → re-serializing → parsing again
    SHALL produce an equivalent object.
    """
    # First serialization
    json_str = response.model_dump_json()

    # First parse
    parsed_once = ExtractionResponse.model_validate_json(json_str)

    # Second serialization
    json_str_2 = parsed_once.model_dump_json()

    # Second parse
    parsed_twice = ExtractionResponse.model_validate_json(json_str_2)

    # Both parsed objects must be equivalent to the original
    assert parsed_once == response
    assert parsed_twice == response
    assert json.loads(json_str) == json.loads(json_str_2)
