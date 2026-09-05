"""Abstract text-to-speech contract implemented by every runtime TTS backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import VoiceProfile


class TTSEngine(ABC):
    """Interface boundary that keeps TTS routing independent of Chatterbox or NeMo internals."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: VoiceProfile | None = None,
        pace: float = 1.0,
    ) -> tuple[bytes, int]:
        """Synthesize normalized text and return WAV bytes plus the waveform sample rate."""
        raise NotImplementedError
