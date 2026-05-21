"""Unit tests for benchmark scorer."""
import pytest
from app.benchmark.scorer import BenchmarkResult, score_sample, SCORED_FIELDS


def make_result() -> BenchmarkResult:
    return BenchmarkResult()


class TestScoreSample:
    def test_all_correct(self):
        result = make_result()
        extracted = {
            "name": "John Doe",
            "document_number": "P1234567",
            "dob": "1990-05-15",
            "expiry": "2030-05-15",
            "country": "USA",
        }
        ground_truth = dict(extracted)
        score_sample(extracted, ground_truth, result)

        assert result.total_samples == 1
        for fname in SCORED_FIELDS:
            assert result.field_scores[fname].correct == 1
            assert result.field_scores[fname].missing == 0
            assert result.field_scores[fname].wrong == 0

    def test_all_missing(self):
        result = make_result()
        extracted = {}
        ground_truth = {
            "name": "John Doe",
            "document_number": "P1234567",
            "dob": "1990-05-15",
            "expiry": "2030-05-15",
            "country": "USA",
        }
        score_sample(extracted, ground_truth, result)

        for fname in SCORED_FIELDS:
            assert result.field_scores[fname].missing == 1
            assert result.field_scores[fname].correct == 0

    def test_partial_correct(self):
        result = make_result()
        extracted = {
            "name": "John Doe",
            "document_number": "WRONG",
            "dob": "1990-05-15",
            "expiry": None,
            "country": "USA",
        }
        ground_truth = {
            "name": "John Doe",
            "document_number": "P1234567",
            "dob": "1990-05-15",
            "expiry": "2030-05-15",
            "country": "USA",
        }
        score_sample(extracted, ground_truth, result)

        assert result.field_scores["name"].correct == 1
        assert result.field_scores["document_number"].wrong == 1
        assert result.field_scores["dob"].correct == 1
        assert result.field_scores["expiry"].missing == 1
        assert result.field_scores["country"].correct == 1

    def test_case_insensitive_comparison(self):
        result = make_result()
        extracted = {"name": "JOHN DOE", "document_number": "p1234567",
                     "dob": "1990-05-15", "expiry": "2030-05-15", "country": "usa"}
        ground_truth = {"name": "john doe", "document_number": "P1234567",
                        "dob": "1990-05-15", "expiry": "2030-05-15", "country": "USA"}
        score_sample(extracted, ground_truth, result)

        for fname in SCORED_FIELDS:
            assert result.field_scores[fname].correct == 1

    def test_accuracy_calculation(self):
        result = make_result()
        gt = {"name": "A", "document_number": "B", "dob": "1990-01-01",
              "expiry": "2030-01-01", "country": "USA"}
        # 3 correct, 1 wrong, 1 missing
        ex = {"name": "A", "document_number": "WRONG", "dob": "1990-01-01",
              "expiry": None, "country": "USA"}
        score_sample(ex, gt, result)

        assert result.field_scores["name"].accuracy == 1.0
        assert result.field_scores["document_number"].accuracy == 0.0
        assert result.field_scores["expiry"].accuracy == 0.0

    def test_multiple_samples_accumulate(self):
        result = make_result()
        gt = {"name": "A", "document_number": "B", "dob": "1990-01-01",
              "expiry": "2030-01-01", "country": "USA"}
        for _ in range(5):
            score_sample(gt, gt, result)

        assert result.total_samples == 5
        for fname in SCORED_FIELDS:
            assert result.field_scores[fname].total == 5
            assert result.field_scores[fname].correct == 5

    def test_overall_accuracy(self):
        result = make_result()
        gt = {"name": "A", "document_number": "B", "dob": "1990-01-01",
              "expiry": "2030-01-01", "country": "USA"}
        score_sample(gt, gt, result)
        assert result.overall_accuracy == 1.0

    def test_summary_keys(self):
        result = make_result()
        summary = result.summary()
        assert "total_samples" in summary
        assert "doc_type_accuracy" in summary
        assert "overall_field_accuracy" in summary
        assert "fields" in summary
        for fname in SCORED_FIELDS:
            assert fname in summary["fields"]
