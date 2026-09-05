"""Regression tests protecting Unicode-safe normalization for every target language."""

from app.normalization.ewe import EweNormalizer
from app.normalization.ga import GaNormalizer
from app.normalization.generic import GenericNormalizer
from app.normalization.hausa import HausaNormalizer
from app.normalization.twi import TwiNormalizer


def test_generic_collapses_whitespace() -> None:
    """Generic normalization should clean spacing without rewriting words."""
    assert GenericNormalizer().normalize("hello   world  !") == "hello world!"


def test_twi_preserves_distinctive_graphemes() -> None:
    """Twi letters and diacritics must survive runtime text sanitation unchanged."""
    text = "Ɛnnɛ yɛbɛkɔ. Ɔbarima no wɔ ha."
    assert TwiNormalizer().normalize(text) == text


def test_ewe_preserves_distinctive_graphemes() -> None:
    """Ewe extended Latin graphemes must not be stripped by an English-centric sanitizer."""
    text = "Ɛ, ƒ, ɖ, ŋ, ɔ le Eʋegbe me."
    assert EweNormalizer().normalize(text) == text


def test_ga_preserves_distinctive_graphemes() -> None:
    """Ga extended letters and ordinary punctuation must remain intact."""
    text = "Ga wiemɔ: ɛ, ŋ, ɔ."
    assert GaNormalizer().normalize(text) == text


def test_hausa_preserves_boko_extended_letters() -> None:
    """Hausa Boko implosives/ejective letters must remain valid training/runtime text."""
    text = "ƙ ɗ ɓ ƴ — Hausa Boko."
    assert HausaNormalizer().normalize(text) == text
