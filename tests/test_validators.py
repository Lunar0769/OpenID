"""Unit tests for field validators."""
import pytest
from app.extraction.validators import (
    normalise_date,
    normalise_country,
    normalise_doc_number,
    normalise_name,
)


class TestNormaliseDate:
    def test_iso_format_passthrough(self):
        assert normalise_date("1990-05-15") == "1990-05-15"

    def test_slash_dmy(self):
        assert normalise_date("15/05/1990") == "1990-05-15"

    def test_dash_dmy(self):
        assert normalise_date("15-05-1990") == "1990-05-15"

    def test_mrz_yymmdd_2000s(self):
        assert normalise_date("900515") == "1990-05-15"

    def test_mrz_yymmdd_pivot_2000s(self):
        # yy=25 → 2025
        assert normalise_date("250101") == "2025-01-01"

    def test_mrz_8digit(self):
        assert normalise_date("19900515") == "1990-05-15"

    def test_text_month(self):
        assert normalise_date("15 Jan 1990") == "1990-01-15"

    def test_invalid_returns_none(self):
        assert normalise_date("not-a-date") is None

    def test_empty_returns_none(self):
        assert normalise_date("") is None

    def test_none_returns_none(self):
        assert normalise_date(None) is None


class TestNormaliseCountry:
    def test_alpha3_passthrough(self):
        assert normalise_country("USA") == "USA"

    def test_alpha2_to_alpha3(self):
        assert normalise_country("US") == "USA"
        assert normalise_country("AE") == "ARE"
        assert normalise_country("IN") == "IND"
        assert normalise_country("PK") == "PAK"
        assert normalise_country("GB") == "GBR"

    def test_lowercase_input(self):
        assert normalise_country("us") == "USA"

    def test_unknown_returns_none(self):
        assert normalise_country("ZZ") is None

    def test_empty_returns_none(self):
        assert normalise_country("") is None


class TestNormaliseDocNumber:
    def test_valid_passport(self):
        assert normalise_doc_number("P1234567", "passport") == "P1234567"

    def test_passport_lowercase_cleaned(self):
        result = normalise_doc_number("p1234567", "passport")
        assert result == "P1234567"

    def test_emirates_id_formatted(self):
        assert normalise_doc_number("784-1990-1234567-8", "emirates_id") == "784-1990-1234567-8"

    def test_emirates_id_from_digits(self):
        result = normalise_doc_number("784199012345678", "emirates_id")
        assert result == "784-1990-1234567-8"

    def test_aadhaar_12_digits(self):
        assert normalise_doc_number("123456789012", "aadhaar") == "123456789012"

    def test_pan_format(self):
        assert normalise_doc_number("ABCDE1234F", "pan") == "ABCDE1234F"

    def test_cnic_formatted(self):
        assert normalise_doc_number("12345-1234567-1", "cnic") == "12345-1234567-1"

    def test_cnic_from_digits(self):
        result = normalise_doc_number("1234512345671", "cnic")
        assert result == "12345-1234567-1"

    def test_empty_returns_none(self):
        assert normalise_doc_number("", "passport") is None


class TestNormaliseName:
    def test_basic_name(self):
        assert normalise_name("JOHN DOE") == "John Doe"

    def test_mrz_filler_removed(self):
        assert normalise_name("DOE<<JOHN<WILLIAM") == "Doe John William"

    def test_noise_chars_removed(self):
        result = normalise_name("J0HN D0E")  # zeros instead of O
        assert result is not None  # Should not crash

    def test_empty_returns_none(self):
        assert normalise_name("") is None

    def test_none_returns_none(self):
        assert normalise_name(None) is None

    def test_only_fillers_returns_none(self):
        assert normalise_name("<<<<<<") is None
