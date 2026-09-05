"""Language-to-engine router that shares general TTS models and isolates custom NeMo checkpoints per language."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.errors import ConfigurationError
from app.engines.tts.base import TTSEngine
from app.engines.tts.chatterbox import ChatterboxTTSEngine
from app.engines.tts.nemo_fastpitch import NemoCheckpointPair, NemoFastPitchTTSEngine
from app.services.language_registry import LanguageRegistry


class TTSRouter:
    """Select a synthesis backend from language configuration while reusing one shared Chatterbox model
    and lazily creating isolated NeMo engines per custom language."""
    def __init__(self, settings: Settings, languages: LanguageRegistry) -> None:
        """Create the cheap routing state only; custom NeMo engines and all model weights remain lazy
        until a language is actually requested."""
        self.settings = settings
        self.languages = languages
        self._shared_chatterbox = ChatterboxTTSEngine(settings)
        self._nemo_by_language: dict[str, NemoFastPitchTTSEngine] = {}

    def engine_for(self, language: str) -> tuple[TTSEngine, str]:
        """Resolve the language specification, return the shared multilingual backend when possible, or
        create/cache the custom NeMo adapter from that language checkpoint pair. Invalid routing
        config fails explicitly."""
        spec = self.languages.get(language)
        if spec.tts_engine == "chatterbox":
            return self._shared_chatterbox, spec.tts_language_id or language
        if spec.tts_engine == "nemo":
            if language not in self._nemo_by_language:
                if not spec.nemo:
                    raise ConfigurationError(f"language {language} is missing its nemo config")
                try:
                    fastpitch = Path(spec.nemo["fastpitch_checkpoint"])
                    hifigan = Path(spec.nemo["hifigan_checkpoint"])
                except KeyError as exc:
                    raise ConfigurationError(f"language {language} has incomplete nemo checkpoints") from exc
                pair = NemoCheckpointPair(
                    fastpitch=fastpitch,
                    hifigan=hifigan,
                    sample_rate=int(spec.nemo.get("sample_rate", 22050)),
                )
                self._nemo_by_language[language] = NemoFastPitchTTSEngine(self.settings, pair)
            return self._nemo_by_language[language], language
        raise ConfigurationError(f"unknown TTS engine {spec.tts_engine!r} for {language}")
