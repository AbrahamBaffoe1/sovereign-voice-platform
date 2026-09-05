"""Regression tests for ASR checkpoint resume and frozen-audio behavior without Transformers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.asr.finetune_whisper import (
    _latest_checkpoint,
    _read_frozen_wav,
    _resolve_resume_checkpoint,
)


def test_latest_checkpoint_uses_highest_numeric_step(tmp_path: Path) -> None:
    """Filesystem ordering must not make checkpoint-900 look newer than checkpoint-1000."""
    output = tmp_path / "hf"
    (output / "checkpoint-250").mkdir(parents=True)
    (output / "checkpoint-1000").mkdir()
    (output / "checkpoint-not-a-step").mkdir()

    assert _latest_checkpoint(output) == output / "checkpoint-1000"
    assert _resolve_resume_checkpoint(output, "auto") == str(output / "checkpoint-1000")


def test_auto_resume_is_clean_when_no_checkpoint_exists(tmp_path: Path) -> None:
    """A first run may use the same --resume command safely; auto simply resolves to no checkpoint yet."""
    assert _resolve_resume_checkpoint(tmp_path / "new-run", "auto") is None


def test_frozen_wav_reader_rechecks_compiler_audio_contract(tmp_path: Path) -> None:
    """Training must reject a corpus file that no longer satisfies the 16 kHz mono freeze contract."""
    mono = tmp_path / "mono.wav"
    sf.write(mono, np.zeros(1600, dtype=np.float32), 16000, subtype="PCM_16")
    waveform = _read_frozen_wav(mono)
    assert waveform.shape == (1600,)

    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((1600, 2), dtype=np.float32), 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="expected mono"):
        _read_frozen_wav(stereo)
