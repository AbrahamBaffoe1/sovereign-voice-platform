"""Typed training-language profiles shared by corpus preparation, ASR experiments and TTS preflight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True, slots=True)
class CorpusPolicy:
    """Dataset invariants that must hold before recordings are admitted to a training split."""
    min_seconds: float
    max_seconds: float
    require_mono: bool
    split_unit: Literal["speaker", "audio"]
    required_metadata: tuple[str, ...]
    require_consent: bool
    require_reviewed_transcript: bool


@dataclass(frozen=True, slots=True)
class ASRTrainingPolicy:
    """Whisper experiment defaults without pretending an unsupported decoder language token exists."""
    base_model: str
    language_token_mode: Literal["none", "explicit"]
    decoder_language: str | None
    sample_rate: int


@dataclass(frozen=True, slots=True)
class TTSTrainingPolicy:
    """TTS frontend/deployment policy used to block training until text-tokenizer review is complete."""
    sample_rate: int
    frontend: Literal["grapheme", "phoneme", "experiment"]
    tokenizer_reviewed: bool
    g2p_reviewed: bool


@dataclass(frozen=True, slots=True)
class LanguageTrainingProfile:
    """One version-controlled source of truth for a language's training and data-governance policy."""
    code: str
    iso639_3: str
    name: str
    aliases: tuple[str, ...]
    script: str
    runtime_normalizer: str
    corpus: CorpusPolicy
    asr: ASRTrainingPolicy
    tts: TTSTrainingPolicy
    reviewed_graphemes: tuple[str, ...] | None
    source_path: Path

    @property
    def tokenizer_ready(self) -> bool:
        """Return true only when the TTS tokenizer and grapheme inventory have both been reviewed."""
        return self.tts.tokenizer_reviewed and bool(self.reviewed_graphemes)


def _mapping(value: Any, field: str, path: Path) -> dict[str, Any]:
    """Require a YAML object at a named field so malformed profiles fail before training starts."""
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {field} must be a mapping")
    return value


def load_language_profile(path: Path) -> LanguageTrainingProfile:
    """Parse and validate one language profile into immutable objects used by training commands."""
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    language = _mapping(payload.get("language"), "language", path)
    corpus = _mapping(payload.get("corpus"), "corpus", path)
    asr = _mapping(payload.get("asr"), "asr", path)
    tts = _mapping(payload.get("tts"), "tts", path)
    code = str(language.get("code", "")).strip()
    iso639_3 = str(language.get("iso639_3", "")).strip()
    name = str(language.get("name", "")).strip()
    script = str(language.get("script", "")).strip()
    normalizer = str(language.get("runtime_normalizer", "")).strip()
    if not all((code, iso639_3, name, script, normalizer)):
        raise ValueError(f"{path}: language code/iso639_3/name/script/runtime_normalizer are required")
    aliases = tuple(str(item).strip() for item in language.get("aliases", []) if str(item).strip())
    required_metadata = tuple(str(item).strip() for item in corpus.get("required_metadata", []))
    split_unit = str(corpus.get("split_unit", "speaker"))
    if split_unit not in {"speaker", "audio"}:
        raise ValueError(f"{path}: corpus.split_unit must be 'speaker' or 'audio'")
    token_mode = str(asr.get("language_token_mode", "none"))
    if token_mode not in {"none", "explicit"}:
        raise ValueError(f"{path}: asr.language_token_mode must be 'none' or 'explicit'")
    decoder_language = asr.get("decoder_language")
    if token_mode == "explicit" and not decoder_language:
        raise ValueError(f"{path}: explicit ASR language token mode requires decoder_language")
    if token_mode == "none" and decoder_language:
        raise ValueError(f"{path}: decoder_language must be null when language_token_mode is none")
    frontend = str(tts.get("frontend", "experiment"))
    if frontend not in {"grapheme", "phoneme", "experiment"}:
        raise ValueError(f"{path}: tts.frontend must be grapheme, phoneme or experiment")
    reviewed = payload.get("reviewed_graphemes")
    reviewed_graphemes = None
    if reviewed is not None:
        if not isinstance(reviewed, list):
            raise ValueError(f"{path}: reviewed_graphemes must be null or a list")
        reviewed_graphemes = tuple(str(item) for item in reviewed)
    return LanguageTrainingProfile(
        code=code,
        iso639_3=iso639_3,
        name=name,
        aliases=aliases,
        script=script,
        runtime_normalizer=normalizer,
        corpus=CorpusPolicy(
            min_seconds=float(corpus.get("min_seconds", 0.5)),
            max_seconds=float(corpus.get("max_seconds", 20.0)),
            require_mono=bool(corpus.get("require_mono", True)),
            split_unit=split_unit,
            required_metadata=required_metadata,
            require_consent=bool(corpus.get("require_consent", True)),
            require_reviewed_transcript=bool(corpus.get("require_reviewed_transcript", True)),
        ),
        asr=ASRTrainingPolicy(
            base_model=str(asr.get("base_model", "openai/whisper-small")),
            language_token_mode=token_mode,
            decoder_language=(str(decoder_language) if decoder_language else None),
            sample_rate=int(asr.get("sample_rate", 16000)),
        ),
        tts=TTSTrainingPolicy(
            sample_rate=int(tts.get("sample_rate", 22050)),
            frontend=frontend,
            tokenizer_reviewed=bool(tts.get("tokenizer_reviewed", False)),
            g2p_reviewed=bool(tts.get("g2p_reviewed", False)),
        ),
        reviewed_graphemes=reviewed_graphemes,
        source_path=path,
    )


def load_profile_directory(directory: Path) -> dict[str, LanguageTrainingProfile]:
    """Load every YAML profile in a directory and reject duplicate canonical codes or aliases."""
    profiles: dict[str, LanguageTrainingProfile] = {}
    aliases: dict[str, str] = {}
    for path in sorted(directory.glob("*.yaml")):
        profile = load_language_profile(path)
        if profile.code in profiles:
            raise ValueError(f"duplicate language profile code: {profile.code}")
        profiles[profile.code] = profile
        for alias in (profile.code, *profile.aliases):
            key = alias.casefold()
            previous = aliases.get(key)
            if previous and previous != profile.code:
                raise ValueError(f"language alias {alias!r} is shared by {previous} and {profile.code}")
            aliases[key] = profile.code
    return profiles
