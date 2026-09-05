"""Abstract text-to-speech contract implemented by every runtime TTS backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import VoiceProfile


class TTSEngine(ABC):
    @abstractmethod
    """Interface boundary for every synthesis backend used by the router."""
    async def synthesize(
        self,
        text: str,
        *,
        language: str,
        voice: VoiceProfile | None = None,
        pace: float = 1.0,
    ) -> tuple[bytes, int]:
        """Return WAV bytes and sample rate."""
        raise NotImplementedError
