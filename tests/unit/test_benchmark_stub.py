"""
Benchmark stub tests.

Task 12.7: Run benchmark suite against labeled dataset and record baseline accuracy scores.

NOTE: This test is skipped because it requires a labeled dataset with ground-truth
document images and expected extraction results. The benchmark suite is implemented
in app/benchmark/ but cannot be run without the dataset.

To run the actual benchmark:
1. Obtain a labeled dataset (CSV with image paths + expected field values)
2. Place it at tests/fixtures/benchmark_dataset.csv
3. Run: python -m app.benchmark.runner --dataset tests/fixtures/benchmark_dataset.csv
4. Record the baseline accuracy scores in this file
"""

import pytest


@pytest.mark.skip(
    reason=(
        "Benchmark requires a labeled dataset with ground-truth document images. "
        "No dataset is available in the test environment. "
        "To run: obtain a labeled dataset and use app/benchmark/runner.py."
    )
)
def test_benchmark_passport_extraction_accuracy():
    """Benchmark: passport extraction field accuracy against labeled dataset.

    Expected metrics to record:
    - name_accuracy: >= 0.90
    - document_number_accuracy: >= 0.95
    - dob_accuracy: >= 0.95
    - expiry_accuracy: >= 0.95
    - country_accuracy: >= 0.98
    - overall_accuracy: >= 0.92
    """
    from app.benchmark.runner import BenchmarkRunner
    from app.benchmark.scorer import score_results

    runner = BenchmarkRunner(dataset_path="tests/fixtures/benchmark_dataset.csv")
    results = runner.run()
    scores = score_results(results)

    assert scores["overall_accuracy"] >= 0.92, (
        f"Overall accuracy {scores['overall_accuracy']:.2%} below 92% baseline"
    )


@pytest.mark.skip(
    reason=(
        "Benchmark requires a labeled dataset with ground-truth document images. "
        "No dataset is available in the test environment."
    )
)
def test_benchmark_id_card_extraction_accuracy():
    """Benchmark: ID card extraction field accuracy against labeled dataset.

    Expected metrics to record:
    - name_accuracy: >= 0.88
    - document_number_accuracy: >= 0.93
    - dob_accuracy: >= 0.93
    - expiry_accuracy: >= 0.93
    - overall_accuracy: >= 0.90
    """
    from app.benchmark.runner import BenchmarkRunner
    from app.benchmark.scorer import score_results

    runner = BenchmarkRunner(
        dataset_path="tests/fixtures/benchmark_dataset.csv",
        doc_type="id_card",
    )
    results = runner.run()
    scores = score_results(results)

    assert scores["overall_accuracy"] >= 0.90, (
        f"Overall accuracy {scores['overall_accuracy']:.2%} below 90% baseline"
    )
