"""Validated, config-driven language catalog used by ASR/TTS routing and API discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import ConfigurationError, UnsupportedLanguageError


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """Immutable runtime description of one language and its model-routing metadata."""

    code: str
    name: str
    iso639_3: str | None
    aliases: tuple[str, ...]
    tts_engine: str
    normalizer: str
    tts_language_id: str | None = None
    training_profile: str | None = None
    asr: dict[str, Any] | None = None
    nemo: dict[str, Any] | None = None


class LanguageRegistry:
    """Load language routing once and canonicalize aliases before any engine sees a request."""

    def __init__(self, config_path: Path) -> None:
        """Parse languages.yaml, build typed specs, and reject duplicate aliases at process startup."""
        if not config_path.exists():
            raise ConfigurationError(f"language config does not exist: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw_languages = payload.get("languages")
        if not isinstance(raw_languages, dict) or not raw_languages:
            raise ConfigurationError("languages.yaml must contain a non-empty 'languages' mapping")

        self._languages: dict[str, LanguageSpec] = {}
        self._aliases: dict[str, str] = {}
        for raw_code, raw in raw_languages.items():
            code = str(raw_code).strip().casefold()
            if not isinstance(raw, dict):
                raise ConfigurationError(f"language {code!r} must be an object")
            aliases = tuple(str(item).strip() for item in raw.get("aliases", []) if str(item).strip())
            spec = LanguageSpec(
                code=code,
                name=str(raw.get("name", code)),
                iso639_3=(str(raw["iso639_3"]) if raw.get("iso639_3") else None),
                aliases=aliases,
                tts_engine=str(raw.get("tts_engine", "chatterbox")),
                normalizer=str(raw.get("normalizer", "generic")),
                tts_language_id=raw.get("tts_language_id"),
                training_profile=raw.get("training_profile"),
                asr=raw.get("asr"),
                nemo=raw.get("nemo"),
            )
            if code in self._languages:
                raise ConfigurationError(f"duplicate language code: {code}")
            self._languages[code] = spec
            for alias in (code, *aliases):
                key = alias.casefold()
                previous = self._aliases.get(key)
                if previous and previous != code:
                    raise ConfigurationError(
                        f"language alias {alias!r} is assigned to both {previous} and {code}"
                    )
                self._aliases[key] = code

    def canonicalize(self, code: str) -> str:
        """Resolve a canonical code from a configured code or human-friendly alias."""
        key = code.strip().casefold()
        try:
            return self._aliases[key]
        except KeyError as exc:
            raise UnsupportedLanguageError(f"unsupported language: {code}") from exc

    def get(self, code: str) -> LanguageSpec:
        """Resolve aliases and return the canonical language specification."""
        return self._languages[self.canonicalize(code)]

    def all(self) -> list[LanguageSpec]:
        """Return canonical language specifications in configuration order for discovery endpoints."""
        return list(self._languages.values())
