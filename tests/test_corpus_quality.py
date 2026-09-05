"""Tests for speaker-disjoint corpus reporting and leakage detection."""

from training.common.corpus_quality import build_quality_report
from training.common.manifest import SpeechRecord, stable_partition


def _record(speaker: str, split: str, dialect: str = "d1") -> SpeechRecord:
    """Create a tiny audit record for corpus quality tests without real audio files."""
    return SpeechRecord(audio_filepath=f"{speaker}-{split}.wav", text="sample", duration=3.0, speaker=speaker, language="tw", dialect=dialect, split=split)


def test_quality_report_detects_speaker_leakage() -> None:
    """A speaker appearing in two splits must be visible instead of inflating held-out accuracy."""
    report = build_quality_report([_record("spk1", "train"), _record("spk1", "test")])
    assert report.speaker_leakage == ["spk1"]


def test_stable_speaker_partition_is_deterministic() -> None:
    """The same speaker id must always resolve to the same split across dataset rebuilds."""
    assert stable_partition("speaker-001") == stable_partition("speaker-001")
