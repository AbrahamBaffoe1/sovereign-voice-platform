"""Tests for the shared training profile and runtime language registries."""

from pathlib import Path

from app.services.language_registry import LanguageRegistry
from training.common.language_profile import load_language_profile, load_profile_directory

ROOT = Path(__file__).resolve().parents[1]


def test_four_target_training_profiles_load() -> None:
    """All target languages should share one validated profile schema rather than bespoke scripts."""
    profiles = load_profile_directory(ROOT / "training/configs/languages")
    assert set(profiles) == {"tw", "gaa", "ee", "ha"}
    assert all(profile.corpus.split_unit == "speaker" for profile in profiles.values())
    assert all(not profile.tokenizer_ready for profile in profiles.values())


def test_runtime_registry_resolves_language_aliases() -> None:
    """Human-friendly names should canonicalize without creating duplicate checkpoint caches."""
    registry = LanguageRegistry(ROOT / "config/languages.yaml")
    assert registry.get("twi").code == "tw"
    assert registry.get("ga").code == "gaa"
    assert registry.get("ewe").code == "ee"
    assert registry.get("hausa").code == "ha"


def test_hausa_profile_keeps_token_strategy_unclaimed() -> None:
    """Do not claim a Whisper decoder token decision before the experiment is reviewed."""
    profile = load_language_profile(ROOT / "training/configs/languages/ha.yaml")
    assert profile.asr.language_token_mode == "none"
    assert profile.asr.decoder_language is None
