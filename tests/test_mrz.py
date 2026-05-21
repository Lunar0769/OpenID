"""Unit tests for MRZ parsing and check digit validation."""
import pytest
from app.extraction.passport import parse_mrz, _mrz_check_digit, _validate_check_digit


class TestMrzCheckDigit:
    def test_known_doc_number(self):
        # ICAO example: "L898902C3" → check digit 6
        assert _mrz_check_digit("L898902C3") == 6

    def test_known_dob(self):
        # "690806" → check digit 1
        assert _mrz_check_digit("690806") == 1

    def test_known_expiry(self):
        # "940623" → check digit 6
        assert _mrz_check_digit("940623") == 6

    def test_filler_chars(self):
        # Fillers ('<') have value 0
        assert _mrz_check_digit("<<<<<<") == 0

    def test_validate_correct(self):
        assert _validate_check_digit("L898902C3", "6") is True

    def test_validate_wrong(self):
        assert _validate_check_digit("L898902C3", "5") is False

    def test_validate_non_digit_check(self):
        assert _validate_check_digit("L898902C3", "X") is False


class TestParseMrz:
    # ICAO Doc 9303 sample passport MRZ
    LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<"
    LINE2 = "L898902C36UTO6908061F9406236ZE184226B<<<<<1"

    def test_parses_name(self):
        result = parse_mrz(self.LINE1, self.LINE2)
        assert result is not None
        assert "eriksson" in result["name"].lower()
        assert "anna" in result["name"].lower()

    def test_parses_doc_number(self):
        result = parse_mrz(self.LINE1, self.LINE2)
        assert result["document_number"] == "L898902C3"

    def test_parses_dob(self):
        result = parse_mrz(self.LINE1, self.LINE2)
        assert result["dob"] == "1969-08-06"

    def test_parses_expiry(self):
        result = parse_mrz(self.LINE1, self.LINE2)
        assert result["expiry"] == "1994-06-23"

    def test_parses_country(self):
        result = parse_mrz(self.LINE1, self.LINE2)
        # UTO is a fictional ICAO code — normalise_country may return None
        # Just check it doesn't crash
        assert "country" in result

    def test_mrz_valid_flag(self):
        result = parse_mrz(self.LINE1, self.LINE2)
        assert result["_mrz_valid"] is True

    def test_invalid_line1_returns_none(self):
        # Line 1 must start with 'P' for passport
        result = parse_mrz("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", self.LINE2)
        assert result is None

    def test_bad_check_digit_marks_invalid(self):
        # Corrupt the doc number check digit
        bad_line2 = "L898902C39UTO6908061F9406236ZE184226B<<<<<1"
        result = parse_mrz(self.LINE1, bad_line2)
        assert result is not None
        assert result["_mrz_valid"] is False
