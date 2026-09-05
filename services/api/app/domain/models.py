"""Transport and domain data structures shared across routes, engines, services, and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AudioEncoding(StrEnum):
    """Names the audio encodings understood at public protocol boundaries."""
    WAV = "wav"
    PCM_S16LE = "pcm_s16le"


class TranscriptionRequestOptions(BaseModel):
    """Validated ASR options that can be reused by future typed endpoints or job queues."""
    language: str | None = None
    hotwords: str | None = None
    word_timestamps: bool = False


class WordTimestamp(BaseModel):
    """One recognized token with model-provided temporal boundaries and optional confidence."""
    word: str
    start: float
    end: float
    probability: float | None = None


class TranscriptionResult(BaseModel):
    """Engine-neutral ASR result returned by HTTP endpoints and consumed by the voice pipeline."""
    text: str
    language: str
    language_probability: float | None = None
    duration_seconds: float | None = None
    words: list[WordTimestamp] = Field(default_factory=list)


class SpeechRequest(BaseModel):
    """Validated text-to-speech request with bounded text length and a deliberately conservative pace
    range."""
    text: str = Field(min_length=1, max_length=10000)
    language: str = "en"
    voice_id: str | None = None
    pace: float = Field(default=1.0, ge=0.5, le=2.0)


class ConversationTurnResult(BaseModel):
    """Serializable metadata for one completed ASR-to-response-to-TTS turn, including stage timings for
    latency diagnosis."""
    transcript: str
    input_language: str
    response_text: str
    output_language: str
    audio_sample_rate: int
    timings_ms: dict[str, float]


class VoiceProfilePublic(BaseModel):
    """Safe projection of a stored voice profile that may be returned to clients."""
    id: str
    name: str
    language: str | None = None
    kind: str
    created_at: str


@dataclass(slots=True)
class VoiceProfile:
    """Internal voice profile containing local reference paths and engine-specific speaker information
    that must not be exposed directly."""
    id: str
    name: str
    language: str | None
    kind: str
    created_at: str
    reference_audio_path: Path | None = None
    nemo_speaker_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
