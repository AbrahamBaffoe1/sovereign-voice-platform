"""Regression tests for corpus-v0 freeze boundaries that must hold before expensive training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data.leakage import assert_no_exact_audio_leakage, exact_audio_leakage_report
from training.prepare_dataset import _assign_split


def _write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_fixed_evaluation_split_overrides_training_only_hint() -> None:
    """A source-level training_only hint must never pull a governed held-out benchmark back into train."""
    split = _assign_split(
        fixed_split="test",
        training_only=True,
        speaker=None,
        digest="a" * 64,
        profile=None,
    )
    assert split == "test"


def test_exact_audio_leakage_report_passes_disjoint_corpora(tmp_path: Path) -> None:
    """Distinct normalized waveforms should produce a machine-readable passing proof."""
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    _write_audit(train, [{"sha256": "a" * 64, "source_id": "train:1"}])
    _write_audit(evaluation, [{"sha256": "b" * 64, "source_id": "eval:1"}])
    report = exact_audio_leakage_report(training_audit=train, evaluation_audit=evaluation)
    assert report["passed"] is True
    assert report["overlap_count"] == 0


def test_exact_audio_leakage_gate_fails_closed_and_writes_report(tmp_path: Path) -> None:
    """Any identical normalized waveform crossing train/eval must abort the freeze and leave evidence."""
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    report_path = tmp_path / "leakage.json"
    digest = "c" * 64
    _write_audit(train, [{"sha256": digest, "source_id": "train:1", "audio_filepath": "/train.wav"}])
    _write_audit(evaluation, [{"sha256": digest, "source_id": "eval:9", "audio_filepath": "/eval.wav"}])
    with pytest.raises(RuntimeError, match="exact audio leakage"):
        assert_no_exact_audio_leakage(
            training_audit=train,
            evaluation_audit=evaluation,
            report_path=report_path,
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["overlap_count"] == 1
    assert payload["overlap_examples"][0]["training_source_ids"] == ["train:1"]
    assert payload["overlap_examples"][0]["evaluation_source_ids"] == ["eval:9"]
