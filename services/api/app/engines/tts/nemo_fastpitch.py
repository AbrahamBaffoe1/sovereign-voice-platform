"""Runtime adapter for custom-language NeMo FastPitch acoustic models paired with HiFi-GAN vocoders."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.core.errors import ConfigurationError, EngineUnavailableError, ModelInferenceError
from app.core.lifecycle import AsyncLazy
from app.domain.models import VoiceProfile
from app.engines.tts.base import TTSEngine
from app.services.audio import float_audio_to_wav_bytes


@dataclass(frozen=True, slots=True)
class NemoCheckpointPair:
    """Immutable description of the acoustic-model/vocoder checkpoint pair required to synthesize one
    custom language."""
    fastpitch: Path
    hifigan: Path
    sample_rate: int = 22050


class NemoFastPitchTTSEngine(TTSEngine):
    """Runtime for custom-language FastPitch + HiFi-GAN `.nemo` checkpoints."""

    def __init__(self, settings: Settings, checkpoints: NemoCheckpointPair) -> None:
        """Bind a specific language checkpoint pair, defer heavyweight restore work, and serialize
        inference against the restored model pair."""
        self.settings = settings
        self.checkpoints = checkpoints
        self._models = AsyncLazy(self._load_models)
        self._inference_lock = asyncio.Semaphore(1)

    async def _load_models(self) -> tuple[Any, Any]:
        """Validate checkpoint existence, resolve the execution device, restore both NeMo models off
        the event loop, and switch them to inference mode."""
        try:
            import torch
            from nemo.collections.tts.models import FastPitchModel, HifiGanModel
        except ImportError as exc:
            raise EngineUnavailableError("NeMo TTS is not installed; install the 'tts-nemo' extra") from exc

        if not self.checkpoints.fastpitch.exists():
            raise ConfigurationError(f"missing FastPitch checkpoint: {self.checkpoints.fastpitch}")
        if not self.checkpoints.hifigan.exists():
            raise ConfigurationError(f"missing HiFi-GAN checkpoint: {self.checkpoints.hifigan}")

        device = self.settings.nemo_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        def load() -> tuple[Any, Any]:
            """Restore FastPitch and HiFi-GAN from local .nemo files, move both to the resolved device,
            and disable training behavior with eval()."""
            fastpitch = FastPitchModel.restore_from(str(self.checkpoints.fastpitch), map_location=device)
            hifigan = HifiGanModel.restore_from(str(self.checkpoints.hifigan), map_location=device)
            fastpitch = fastpitch.to(device).eval()
            hifigan = hifigan.to(device).eval()
            return fastpitch, hifigan

        return await asyncio.to_thread(load)

    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: VoiceProfile | None = None,
        pace: float = 1.0,
    ) -> tuple[bytes, int]:
        """Parse language text with the checkpoint tokenizer, generate a mel spectrogram with
        FastPitch, vocode it with HiFi-GAN, and return normalized WAV bytes at the checkpoint sample
        rate."""
        del language
        fastpitch, hifigan = await self._models.get()

        def infer() -> tuple[bytes, int]:
            """Run the synchronous two-stage NeMo forward path under torch.inference_mode so autograd
            state is not allocated during serving."""
            import torch

            with torch.inference_mode():
                parsed = fastpitch.parse(text)
                speaker_id = voice.nemo_speaker_id if voice else None
                spectrogram = fastpitch.generate_spectrogram(
                    tokens=parsed,
                    speaker=speaker_id,
                    pace=pace,
                )
                waveform = hifigan.convert_spectrogram_to_audio(spec=spectrogram)
            audio = waveform.detach().float().cpu().numpy().squeeze()
            return (
                float_audio_to_wav_bytes(np.asarray(audio), self.checkpoints.sample_rate),
                self.checkpoints.sample_rate,
            )

        try:
            async with self._inference_lock:
                return await asyncio.to_thread(infer)
        except Exception as exc:
            if isinstance(exc, (ConfigurationError, EngineUnavailableError)):
                raise
            raise ModelInferenceError(f"NeMo TTS failed: {exc}") from exc
