"""VoxCPM2 TTS adapter for adapted multilingual checkpoints and reference-voice conditioning."""

from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.core.errors import EngineUnavailableError, ModelInferenceError
from app.core.lifecycle import AsyncLazy
from app.domain.models import VoiceProfile
from app.engines.tts.base import TTSEngine


class VoxCPM2TTSEngine(TTSEngine):
    """Serve one explicit checkpoint without claiming the unadapted base supports a target language."""

    def __init__(self, settings: Settings, checkpoint: str | Path) -> None:
        """Capture checkpoint identity and defer expensive model construction until first synthesis."""
        self.settings = settings
        self.checkpoint = str(checkpoint)
        self._model: AsyncLazy[Any] = AsyncLazy(self._load_model)

    async def _load_model(self) -> Any:
        """Import VoxCPM on demand and load the configured local/adapted checkpoint off the event loop."""
        try:
            from voxcpm import VoxCPM
        except ImportError as exc:
            raise EngineUnavailableError("VoxCPM is not installed; install the 'tts-voxcpm' extra") from exc

        def load() -> Any:
            return VoxCPM.from_pretrained(
                hf_model_id=self.checkpoint,
                load_denoiser=False,
                optimize=True,
                device=self.settings.voxcpm_device,
                local_files_only=Path(self.checkpoint).exists(),
            )

        return await asyncio.to_thread(load)

    @staticmethod
    def _to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """Convert model float output to mono PCM16 WAV bytes used by the platform TTS contract."""
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return buffer.getvalue()

    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: VoiceProfile | None = None,
        pace: float = 1.0,
    ) -> tuple[bytes, int]:
        """Generate speech from an adapted checkpoint and optionally condition on reference audio."""
        del language
        if not 0.5 <= pace <= 2.0:
            raise ModelInferenceError("VoxCPM pace must be between 0.5 and 2.0")
        model = await self._model.get()
        reference = str(voice.reference_audio_path) if voice and voice.reference_audio_path else None

        def infer() -> tuple[bytes, int]:
            try:
                kwargs: dict[str, Any] = {
                    "text": text,
                    "cfg_value": self.settings.voxcpm_cfg_value,
                    "inference_timesteps": self.settings.voxcpm_inference_timesteps,
                    "denoise": False,
                }
                if reference:
                    kwargs["prompt_wav_path"] = reference
                samples = model.generate(**kwargs)
                sample_rate = int(model.tts_model.sample_rate)
                return self._to_wav(np.asarray(samples), sample_rate), sample_rate
            except Exception as exc:
                raise ModelInferenceError(f"VoxCPM2 synthesis failed: {exc}") from exc

        return await asyncio.to_thread(infer)
