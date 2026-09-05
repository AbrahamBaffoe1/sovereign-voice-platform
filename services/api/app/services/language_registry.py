"""Validated, config-driven language catalog used by API discovery and TTS routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import ConfigurationError, UnsupportedLanguageError


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """Immutable runtime view of one language route, including its normalizer, TTS backend, and
    optional custom NeMo checkpoint metadata."""
    code: str
    name: str
    tts_engine: str
    normalizer: str
    tts_language_id: str | None = None
    nemo: dict[str, Any] | None = None


class LanguageRegistry:
    """Load and validate languages.yaml once so downstream code consumes typed language records instead
    of repeatedly parsing untrusted dictionaries."""
    def __init__(self, config_path: Path) -> None:
        """Parse the configured YAML file, require a non-empty language mapping, and convert each entry
        into a typed LanguageSpec used for the lifetime of the process."""
        if not config_path.exists():
            raise ConfigurationError(f"language config does not exist: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw_languages = payload.get("languages")
        if not isinstance(raw_languages, dict) or not raw_languages:
            raise ConfigurationError("languages.yaml must contain a non-empty 'languages' mapping")
        self._languages: dict[str, LanguageSpec] = {}
        for code, raw in raw_languages.items():
            if not isinstance(raw, dict):
                raise ConfigurationError(f"language {code!r} must be an object")
            self._languages[code] = LanguageSpec(
                code=code,
                name=str(raw.get("name", code)),
                tts_engine=str(raw.get("tts_engine", "chatterbox")),
                normalizer=str(raw.get("normalizer", "generic")),
                tts_language_id=raw.get("tts_language_id"),
                nemo=raw.get("nemo"),
            )

    def get(self, code: str) -> LanguageSpec:
        """Resolve a language code or raise a domain-specific UnsupportedLanguageError that transports
        can map consistently."""
        try:
            return self._languages[code]
        except KeyError as exc:
            raise UnsupportedLanguageError(f"unsupported language: {code}") from exc

    def all(self) -> list[LanguageSpec]:
        """Return configured language specifications in configuration order for discovery/readiness
        responses."""
        return list(self._languages.values())
