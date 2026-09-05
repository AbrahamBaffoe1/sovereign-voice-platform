"""Regression tests that protect language-safe normalization behavior."""

from app.normalization.generic import GenericNormalizer
from app.normalization.twi import TwiNormalizer


def test_generic_collapses_whitespace() -> None:
    """Regression test that verifies generic collapses whitespace. It protects this behavior from
    silent changes during refactors."""
    assert GenericNormalizer().normalize("hello   world  !") == "hello world!"


def test_twi_preserves_distinctive_graphemes() -> None:
    """Regression test that verifies twi preserves distinctive graphemes. It protects this behavior
    from silent changes during refactors."""
    text = "Ɛnnɛ yɛbɛkɔ. Ɔbarima no wɔ ha."
    assert TwiNormalizer().normalize(text) == text
