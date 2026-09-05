"""Tests for language-aware ASR routing without loading heavyweight speech models."""

from pathlib import Path

from app.config import Settings
from app.services.asr_router import ASRRouter
from app.services.language_registry import LanguageRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_target_languages_are_custom_asr_routes() -> None:
    """The four training targets must not silently route through the generic shared model."""
    settings = Settings(language_config=ROOT / "config/languages.yaml")
    router = ASRRouter(settings, LanguageRegistry(settings.language_config))
    assert router.route_description("twi")["mode"] == "custom"
    assert router.route_description("ga")["model"] == "models/asr/gaa"
    assert router.route_description("ewe")["language"] == "ee"
    assert router.route_description("hausa")["language"] == "ha"


def test_default_auto_detect_route_remains_shared() -> None:
    """No explicit language hint should keep using the deployment's generic discovery model."""
    settings = Settings(language_config=ROOT / "config/languages.yaml", asr_model="large-v3")
    router = ASRRouter(settings, LanguageRegistry(settings.language_config))
    assert router.route_description(None) == {"language": None, "mode": "shared", "model": "large-v3"}
