"""Abstract speech-recognition contract used by the orchestration layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import TranscriptionResult


class ASREngine(ABC):
    @abstractmethod
    """Interface boundary for speech recognition. The pipeline depends on this contract so
    Faster-Whisper can be replaced or mocked without changing application logic."""
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        hotwords: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Convert encoded audio bytes into normalized transcription metadata. Implementations own
        decoding, model execution, and model-specific error translation."""
        raise NotImplementedError
