"""Audio normalization and conservative speech segmentation for corpus intake."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.errors import InvalidAudioError


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One normalized mono PCM16 WAV segment with timing provenance into its parent recording."""

    wav_bytes: bytes
    start_seconds: float
    end_seconds: float
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        """Return the exact segment duration represented by the parent timing boundaries."""
        return self.end_seconds - self.start_seconds


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Serialize normalized float samples to a deterministic mono PCM16 WAV container."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def _decode_with_soundfile(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode formats supported by libsndfile and collapse channels without changing sample rate."""
    with sf.SoundFile(io.BytesIO(payload)) as source:
        samples = source.read(dtype="float32", always_2d=True)
        sample_rate = int(source.samplerate)
    if samples.size == 0:
        raise InvalidAudioError("audio contains no samples")
    mono = samples.mean(axis=1, dtype=np.float32)
    return mono, sample_rate


def _decode_with_ffmpeg(payload: bytes, *, max_seconds: float, target_rate: int) -> tuple[np.ndarray, int]:
    """Use ffmpeg only as a bounded decoder fallback for common phone/voice-message containers."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise InvalidAudioError("audio format requires ffmpeg, but ffmpeg is not installed")
    with tempfile.TemporaryDirectory(prefix="voice-corpus-") as directory:
        source = Path(directory) / "input.bin"
        output = Path(directory) / "output.wav"
        source.write_bytes(payload)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-t",
            str(max_seconds),
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, timeout=max(30.0, max_seconds * 2.0), capture_output=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise InvalidAudioError("ffmpeg could not decode the uploaded recording") from exc
        return _decode_with_soundfile(output.read_bytes())


def decode_audio(payload: bytes, *, max_seconds: float, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Decode uploaded audio, enforce a hard duration cap, and resample to the requested corpus rate."""
    if not payload:
        raise InvalidAudioError("audio upload is empty")
    try:
        samples, sample_rate = _decode_with_soundfile(payload)
    except (RuntimeError, ValueError, sf.LibsndfileError):
        samples, sample_rate = _decode_with_ffmpeg(payload, max_seconds=max_seconds, target_rate=target_rate)

    duration = len(samples) / sample_rate
    if duration > max_seconds + 0.05:
        raise InvalidAudioError(f"recording exceeds maximum duration of {max_seconds:g}s")
    if sample_rate == target_rate:
        return samples.astype(np.float32, copy=False), target_rate

    target_length = max(1, round(len(samples) * target_rate / sample_rate))
    old_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(new_positions, old_positions, samples).astype(np.float32)
    return resampled, target_rate


def normalize_clip(payload: bytes, *, max_seconds: float, target_rate: int = 16000) -> AudioSegment:
    """Normalize one already-cut utterance without applying speech segmentation."""
    samples, sample_rate = decode_audio(payload, max_seconds=max_seconds, target_rate=target_rate)
    duration = len(samples) / sample_rate
    return AudioSegment(_wav_bytes(samples, sample_rate), 0.0, duration, sample_rate)


def segment_recording(
    payload: bytes,
    *,
    max_seconds: float,
    target_rate: int = 16000,
    min_segment_seconds: float = 0.6,
    max_segment_seconds: float = 20.0,
    trailing_silence_seconds: float = 0.45,
) -> list[AudioSegment]:
    """Split a long single-speaker recording using energy-based speech regions with deterministic bounds."""
    samples, sample_rate = decode_audio(payload, max_seconds=max_seconds, target_rate=target_rate)
    frame_seconds = 0.02
    frame = max(1, round(frame_seconds * sample_rate))
    frame_count = (len(samples) + frame - 1) // frame
    energies = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        block = samples[index * frame : (index + 1) * frame]
        if block.size:
            energies[index] = float(np.sqrt(np.mean(block * block) + 1e-12))

    floor = float(np.percentile(energies, 20)) if energies.size else 0.0
    peak = float(np.percentile(energies, 95)) if energies.size else 0.0
    threshold = max(0.003, floor * 2.5, peak * 0.08)
    voiced = energies >= threshold
    if not np.any(voiced):
        return []

    silence_frames = max(1, round(trailing_silence_seconds / frame_seconds))
    max_frames = max(1, round(max_segment_seconds / frame_seconds))
    min_frames = max(1, round(min_segment_seconds / frame_seconds))

    regions: list[tuple[int, int]] = []
    start: int | None = None
    silence_run = 0
    for index, is_voiced in enumerate(voiced):
        if is_voiced:
            if start is None:
                start = index
            silence_run = 0
        elif start is not None:
            silence_run += 1
            if silence_run >= silence_frames:
                end = index - silence_run + 1
                if end - start >= min_frames:
                    regions.append((start, end))
                start = None
                silence_run = 0
        if start is not None and index - start + 1 >= max_frames:
            regions.append((start, index + 1))
            start = None
            silence_run = 0
    if start is not None:
        end = len(voiced)
        if end - start >= min_frames:
            regions.append((start, end))

    segments: list[AudioSegment] = []
    pad = round(0.08 * sample_rate)
    for start_frame, end_frame in regions:
        start_sample = max(0, start_frame * frame - pad)
        end_sample = min(len(samples), end_frame * frame + pad)
        segment = samples[start_sample:end_sample]
        if not segment.size:
            continue
        start_seconds = start_sample / sample_rate
        end_seconds = end_sample / sample_rate
        segments.append(AudioSegment(_wav_bytes(segment, sample_rate), start_seconds, end_seconds, sample_rate))
    return segments
