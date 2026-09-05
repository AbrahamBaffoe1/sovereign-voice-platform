"""Abstract speech-recognition contract used by the orchestration layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import TranscriptionResult


class ASREngine(ABC):
    """Interface boundary for speech recognition used by orchestration and routing code."""

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        hotwords: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Convert encoded audio bytes into normalized transcription metadata.

        Implementations own decoding, model execution, and model-specific error translation so the
        rest of the application never depends directly on Faster-Whisper or another ASR library.
        """
        raise NotImplementedError
