"""Audio boundary utilities for validating, decoding, converting, and packaging waveform data."""

from __future__ import annotations

import io
import wave

import numpy as np
import soundfile as sf

from app.core.errors import InvalidAudioError


def pcm16_mono_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw little-endian mono PCM16 microphone bytes in a valid WAV container after checking the
    sample-rate and 16-bit byte-alignment invariants expected by downstream decoders."""
    if sample_rate < 8000 or sample_rate > 48000:
        raise InvalidAudioError(f"unsupported sample rate: {sample_rate}")
    if len(pcm) % 2:
        raise InvalidAudioError("PCM16 payload has an odd byte length")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()


def float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Validate a synthesized floating-point waveform, peak-normalize only when it exceeds the legal
    [-1, 1] range, then encode deterministic mono PCM16 WAV bytes."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if audio.ndim != 1:
        raise InvalidAudioError("TTS engine produced non-mono audio")
    if not np.all(np.isfinite(audio)):
        raise InvalidAudioError("TTS engine produced NaN/Inf audio")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    out = io.BytesIO()
    sf.write(out, audio, sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def pcm_duration_seconds(byte_count: int, sample_rate: int, channels: int = 1) -> float:
    """Compute raw PCM duration from byte count without decoding audio; WebSocket buffering uses this
    cheap calculation on every incoming frame."""
    bytes_per_sample = 2
    return byte_count / float(sample_rate * channels * bytes_per_sample)


def normalize_reference_audio_to_wav(audio_bytes: bytes, *, max_seconds: float = 30.0) -> bytes:
    """Decode a libsndfile-supported reference clip and return mono PCM16 WAV.

    Reference clips are not blindly stored with a `.wav` suffix. This prevents an
    invalid/mislabeled upload from surfacing later as an opaque TTS model failure.
    """
    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception as exc:
        raise InvalidAudioError(
            "voice reference must be a readable WAV/FLAC/OGG audio file"
        ) from exc
    if data.size == 0:
        raise InvalidAudioError("voice reference is empty")
    duration = data.shape[0] / float(sample_rate)
    if duration < 1.0:
        raise InvalidAudioError("voice reference must be at least 1 second")
    if duration > max_seconds:
        raise InvalidAudioError(f"voice reference must be <= {max_seconds:g} seconds")
    mono = data.mean(axis=1, dtype=np.float32)
    return float_audio_to_wav_bytes(mono, int(sample_rate))
