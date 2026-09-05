"""Deterministic audio-quality measurements used to reject obviously bad training samples before model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True, slots=True)
class AudioQuality:
    """Measured recording properties that can be logged, thresholded, and reviewed without rerunning
    audio decoding."""
    sample_rate: int
    channels: int
    duration: float
    peak: float
    rms: float
    clipped_fraction: float
    dc_offset: float

    @property
    def suspicious(self) -> bool:
        """Apply deliberately conservative heuristic thresholds for clearly risky training audio. This
        is a screening signal, not a substitute for listening/native-speaker QA."""
        return (
            self.duration < 0.35
            or self.duration > 30.0
            or self.rms < 1e-4
            or self.clipped_fraction > 0.005
            or abs(self.dc_offset) > 0.05
        )


def inspect_audio(path: Path) -> AudioQuality:
    """Decode a recording, downmix only for measurement, and calculate duration, peak, RMS, clipping
    ratio, and DC offset used by dataset acceptance rules."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError(f"empty audio file: {path}")
    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    clipped = float(np.mean(np.abs(mono) >= 0.999))
    dc_offset = float(np.mean(mono))
    duration = len(mono) / sample_rate
    return AudioQuality(sample_rate=int(sample_rate),channels=int(data.shape[1]),duration=float(duration),peak=peak,rms=rms,clipped_fraction=clipped,dc_offset=dc_offset)
