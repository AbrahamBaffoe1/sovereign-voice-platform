"""Language-aware ASR router for shared Whisper and per-language fine-tuned checkpoints."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.errors import ConfigurationError, EngineUnavailableError
from app.domain.models import TranscriptionResult
from app.engines.asr.base import ASREngine
from app.engines.asr.faster_whisper import FasterWhisperEngine
from app.services.language_registry import LanguageRegistry, LanguageSpec


class ASRRouter(ASREngine):
    """Route transcription to a shared model or a language-specific fine-tuned checkpoint."""

    def __init__(self, settings: Settings, languages: LanguageRegistry) -> None:
        """Create cheap routing state; heavyweight checkpoints remain lazy inside engine instances."""
        self.settings = settings
        self.languages = languages
        self._shared = FasterWhisperEngine(settings)
        self._custom: dict[str, FasterWhisperEngine] = {}

    def _route(self, language: str | None) -> tuple[FasterWhisperEngine, LanguageSpec | None, str | None]:
        """Resolve an engine plus the actual Whisper language hint to send to that checkpoint."""
        if language is None:
            return self._shared, None, None
        spec = self.languages.get(language)
        asr = spec.asr or {"mode": "shared"}
        mode = str(asr.get("mode", "shared"))
        runtime_hint = asr.get("runtime_language_hint")
        hint = str(runtime_hint) if runtime_hint else None
        if mode == "shared":
            return self._shared, spec, hint or spec.code
        if mode != "custom":
            raise ConfigurationError(f"unknown ASR mode {mode!r} for {spec.code}")
        model = str(asr.get("model", "")).strip()
        if not model:
            raise ConfigurationError(f"custom ASR language {spec.code} is missing asr.model")
        model_kind = str(asr.get("model_kind", "local"))
        if model_kind == "local" and not Path(model).exists():
            raise EngineUnavailableError(
                f"custom ASR checkpoint for {spec.name} is not deployed at {model}; "
                f"train/export it before enabling {spec.code} transcription"
            )
        if model_kind not in {"local", "remote"}:
            raise ConfigurationError(f"unknown asr.model_kind {model_kind!r} for {spec.code}")
        engine = self._custom.get(spec.code)
        if engine is None:
            engine = FasterWhisperEngine(self.settings, model_name=model)
            self._custom[spec.code] = engine
        return engine, spec, hint

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        hotwords: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe with the routed checkpoint and canonicalize explicit custom-language results."""
        engine, spec, hint = self._route(language)
        result = await engine.transcribe(
            audio_bytes,
            language=hint,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
        )
        if spec is None:
            return result
        asr = spec.asr or {}
        if str(asr.get("mode", "shared")) == "custom":
            return result.model_copy(update={"language": spec.code, "language_probability": None})
        return result.model_copy(update={"language": spec.code})

    def route_description(self, language: str | None) -> dict[str, str | None]:
        """Describe routing without loading models; useful for diagnostics, tests and deployment UIs."""
        if language is None:
            return {"language": None, "mode": "shared", "model": self.settings.asr_model}
        spec = self.languages.get(language)
        asr = spec.asr or {"mode": "shared"}
        return {
            "language": spec.code,
            "mode": str(asr.get("mode", "shared")),
            "model": str(asr.get("model", self.settings.asr_model)),
        }
