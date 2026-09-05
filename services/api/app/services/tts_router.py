"""Language-to-engine router sharing general TTS while isolating custom checkpoints per language."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.errors import ConfigurationError, EngineUnavailableError
from app.engines.tts.base import TTSEngine
from app.engines.tts.chatterbox import ChatterboxTTSEngine
from app.engines.tts.nemo_fastpitch import NemoCheckpointPair, NemoFastPitchTTSEngine
from app.engines.tts.voxcpm2 import VoxCPM2TTSEngine
from app.services.language_registry import LanguageRegistry


class TTSRouter:
    """Select synthesis backends from canonical language configuration with lazy model construction."""

    def __init__(self, settings: Settings, languages: LanguageRegistry) -> None:
        """Build routing state only; heavyweight model weights still load on first inference."""
        self.settings = settings
        self.languages = languages
        self._shared_chatterbox = ChatterboxTTSEngine(settings)
        self._nemo_by_language: dict[str, NemoFastPitchTTSEngine] = {}
        self._voxcpm_by_language: dict[str, VoxCPM2TTSEngine] = {}

    def engine_for(self, language: str) -> tuple[TTSEngine, str]:
        """Return the configured engine and canonical engine-language identifier."""
        spec = self.languages.get(language)
        if spec.tts_engine == "chatterbox":
            return self._shared_chatterbox, spec.tts_language_id or spec.code
        if spec.tts_engine == "nemo":
            if spec.code not in self._nemo_by_language:
                if not spec.nemo:
                    raise ConfigurationError(f"language {spec.code} is missing its nemo config")
                try:
                    fastpitch = Path(spec.nemo["fastpitch_checkpoint"])
                    hifigan = Path(spec.nemo["hifigan_checkpoint"])
                except KeyError as exc:
                    raise ConfigurationError(f"language {spec.code} has incomplete nemo checkpoints") from exc
                missing = [str(path) for path in (fastpitch, hifigan) if not path.exists()]
                if missing:
                    raise EngineUnavailableError(
                        f"custom TTS checkpoint(s) for {spec.name} are not deployed: {missing}"
                    )
                pair = NemoCheckpointPair(
                    fastpitch=fastpitch,
                    hifigan=hifigan,
                    sample_rate=int(spec.nemo.get("sample_rate", 22050)),
                )
                self._nemo_by_language[spec.code] = NemoFastPitchTTSEngine(self.settings, pair)
            return self._nemo_by_language[spec.code], spec.code
        if spec.tts_engine == "voxcpm":
            if not spec.voxcpm or not str(spec.voxcpm.get("checkpoint", "")).strip():
                raise ConfigurationError(f"language {spec.code} is missing its adapted VoxCPM checkpoint")
            checkpoint = str(spec.voxcpm["checkpoint"])
            if bool(spec.voxcpm.get("local_only", True)) and not Path(checkpoint).exists():
                raise EngineUnavailableError(
                    f"adapted VoxCPM checkpoint for {spec.name} is not deployed at {checkpoint}"
                )
            engine = self._voxcpm_by_language.get(spec.code)
            if engine is None:
                engine = VoxCPM2TTSEngine(self.settings, checkpoint)
                self._voxcpm_by_language[spec.code] = engine
            return engine, spec.code
        raise ConfigurationError(f"unknown TTS engine {spec.tts_engine!r} for {spec.code}")
