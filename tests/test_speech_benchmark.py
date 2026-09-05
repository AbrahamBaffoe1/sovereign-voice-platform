"""Accuracy/latency benchmark tests for speech-model regression gates."""

import pytest

from training.benchmarks.speech import BenchmarkUtterance, benchmark_report, error_rate


def test_word_error_rate_counts_edits() -> None:
    """WER numerators and denominators should be auditable rather than averaged percentages."""
    edits, words = error_rate("hello world", "hello there", unit="word")
    assert edits == 1
    assert words == 2


def test_report_slices_and_real_time_factor() -> None:
    """A report should expose dialect failures and serving speed next to global accuracy."""
    rows = [
        BenchmarkUtterance("a b", "a b", 2.0, 0.5, dialect="d1"),
        BenchmarkUtterance("a b", "a x", 2.0, 0.5, dialect="d2"),
    ]
    report = benchmark_report(rows)
    assert report["real_time_factor"] == pytest.approx(0.25)
    assert "d1" in report["slices"]["dialect"]
    assert "d2" in report["slices"]["dialect"]
