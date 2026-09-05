"""Chatterbox Multilingual TTS adapter with local model loading and optional reference-voice prompting."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from app.config import Settings
from app.core.errors import EngineUnavailableError, ModelInferenceError
from app.core.lifecycle import AsyncLazy
from app.domain.models import VoiceProfile
from app.engines.tts.base import TTSEngine
from app.services.audio import float_audio_to_wav_bytes


class ChatterboxTTSEngine(TTSEngine):
    """Shared multilingual Chatterbox adapter. One lazily loaded model can serve several configured
    languages while reference-audio prompting remains per request."""
    def __init__(self, settings: Settings) -> None:
        """Prepare lazy Chatterbox loading and serialize access to the shared model instance to keep
        device memory usage predictable."""
        self.settings = settings
        self._model = AsyncLazy(self._load_model)
        self._inference_lock = asyncio.Semaphore(1)

    async def _load_model(self) -> Any:
        """Import optional TTS dependencies at first use, resolve auto device selection, and load the
        pretrained model in a worker thread rather than blocking the event loop."""
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:
            raise EngineUnavailableError(
                "Chatterbox is not installed; install the 'tts-chatterbox' extra"
            ) from exc

        device = self.settings.chatterbox_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        def load() -> Any:
            """Construct the synchronous Chatterbox model with the configured multilingual checkpoint
            on the resolved accelerator."""
            return ChatterboxMultilingualTTS.from_pretrained(
                device=device,
                t3_model=self.settings.chatterbox_model,
            )

        return await asyncio.to_thread(load)

    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: VoiceProfile | None = None,
        pace: float = 1.0,
    ) -> tuple[bytes, int]:
        # Chatterbox does not expose a general pace parameter in the stable multilingual API.
        # We deliberately refuse to fake it with waveform time-stretching because that can
        # damage quality. Pace remains part of the common interface for NeMo/custom engines.
        """Generate speech locally, optionally condition on an enrolled reference voice, normalize
        model output into a mono float array, encode it as PCM16 WAV, and translate model failures
        into a stable domain error."""
        del pace
        model = await self._model.get()

        def infer() -> tuple[bytes, int]:
            """Run synchronous Chatterbox generation and move tensor output back to CPU before waveform
            packaging."""
            kwargs: dict[str, Any] = {"language_id": language}
            if voice and voice.reference_audio_path:
                kwargs["audio_prompt_path"] = str(voice.reference_audio_path)
            wav = model.generate(text, **kwargs)
            if hasattr(wav, "detach"):
                wav = wav.detach().float().cpu().numpy()
            audio = np.asarray(wav, dtype=np.float32).squeeze()
            sample_rate = int(model.sr)
            return float_audio_to_wav_bytes(audio, sample_rate), sample_rate

        try:
            async with self._inference_lock:
                return await asyncio.to_thread(infer)
        except Exception as exc:
            if isinstance(exc, EngineUnavailableError):
                raise
            raise ModelInferenceError(f"Chatterbox TTS failed: {exc}") from exc
