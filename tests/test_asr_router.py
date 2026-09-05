"""Tests for language-aware ASR routing without loading heavyweight speech models."""

from pathlib import Path

from app.config import Settings
from app.services.asr_router import ASRRouter
from app.services.language_registry import LanguageRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_target_languages_are_custom_asr_routes() -> None:
    """The four training targets must use explicit production pointers, never the generic shared model."""
    settings = Settings(language_config=ROOT / "config/languages.yaml")
    router = ASRRouter(settings, LanguageRegistry(settings.language_config))

    expected = {
        "twi": ("tw", "models/deployments/asr/tw/production.json"),
        "ga": ("gaa", "models/deployments/asr/gaa/production.json"),
        "ewe": ("ee", "models/deployments/asr/ee/production.json"),
        "hausa": ("ha", "models/deployments/asr/ha/production.json"),
    }
    for alias, (language, model) in expected.items():
        route = router.route_description(alias)
        assert route["mode"] == "custom"
        assert route["language"] == language
        assert route["model"] == model


def test_default_auto_detect_route_remains_shared() -> None:
    """No explicit language hint should keep using the deployment's generic discovery model."""
    settings = Settings(language_config=ROOT / "config/languages.yaml", asr_model="large-v3")
    router = ASRRouter(settings, LanguageRegistry(settings.language_config))
    assert router.route_description(None) == {
        "language": None,
        "mode": "shared",
        "model": "large-v3",
    }
