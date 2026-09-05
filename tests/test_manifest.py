"""Regression tests for transcript normalization and deterministic dataset partitioning."""

from training.common.manifest import normalize_transcript, stable_partition


def test_normalize_transcript_is_stable() -> None:
    """Regression test that verifies normalize transcript is stable. It protects this behavior from
    silent changes during refactors."""
    assert normalize_transcript("  hello\u00a0  world ") == "hello world"


def test_partition_is_deterministic() -> None:
    """Regression test that verifies partition is deterministic. It protects this behavior from silent
    changes during refactors."""
    assert stable_partition("abc") == stable_partition("abc")
    assert stable_partition("abc") in {"train", "validation", "test"}
