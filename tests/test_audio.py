"""Regression tests for audio conversion and reference-audio normalization."""

import io
import wave

import numpy as np
import soundfile as sf

from app.services.audio import (
    normalize_reference_audio_to_wav,
    pcm16_mono_to_wav_bytes,
    pcm_duration_seconds,
)


def test_pcm_duration() -> None:
    """Regression test that verifies pcm duration. It protects this behavior from silent changes during
    refactors."""
    assert pcm_duration_seconds(32000, 16000) == 1.0


def test_pcm_to_wav_header() -> None:
    """Regression test that verifies pcm to wav header. It protects this behavior from silent changes
    during refactors."""
    payload = b"\x00\x00" * 16000
    wav_bytes = pcm16_mono_to_wav_bytes(payload, 16000)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 16000


def test_reference_audio_is_normalized_to_wav() -> None:
    """Regression test that verifies reference audio is normalized to wav. It protects this behavior
    from silent changes during refactors."""
    source = io.BytesIO()
    sf.write(source, np.zeros(16000, dtype=np.float32), 16000, format="FLAC")
    output = normalize_reference_audio_to_wav(source.getvalue())
    with wave.open(io.BytesIO(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
