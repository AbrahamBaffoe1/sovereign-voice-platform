"""Tests for TTS readiness evidence and production blockers without loading NeMo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data.bootstrap import _read_version
from training.tts.preflight import build_readiness_report

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_tts_artifact(root: Path, *, language: str = "tw") -> Path:
    corpus = root / language / "tts" / "corpus-v0"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "dataset_version.json").write_text(
        json.dumps(
            {
                "language": language,
                "dataset_id": f"{language}-stub",
                "fingerprint_sha256": "a" * 64,
                "accepted": 2,
                "hours": 0.001,
            }
        ),
        encoding="utf-8",
    )
    (corpus / "inventory.json").write_text(
        json.dumps(
            {
                "language": language,
                "accepted": 2,
                "rejected": 0,
                "characters": [
                    {"char": "a", "count": 3},
                    {"char": "ɛ", "count": 2},
                    {"char": " ", "count": 1},
                    {"char": ".", "count": 1},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (corpus / "quality_report.json").write_text(
        json.dumps({"hours": 0.001, "speakers": 2, "speaker_leakage": []}),
        encoding="utf-8",
    )
    _write_jsonl(
        corpus / "audit.jsonl",
        [
            {
                "audio_filepath": "/a.wav",
                "text": "a",
                "duration": 1.0,
                "speaker": "s1",
                "source_id": "waxal_twi_tts:1",
                "source_license": "CC-BY-4.0",
                "source_revision": "1" * 40,
                "sha256": "1" * 64,
                "split": "train",
            },
            {
                "audio_filepath": "/b.wav",
                "text": "ɛ",
                "duration": 1.0,
                "speaker": "s2",
                "source_id": "waxal_twi_tts:2",
                "source_license": "CC-BY-4.0",
                "source_revision": "1" * 40,
                "sha256": "2" * 64,
                "split": "validation",
            },
        ],
    )
    _write_jsonl(corpus / "train.json", [{"audio_filepath": "/a.wav", "text": "a"}])
    _write_jsonl(corpus / "validation.json", [{"audio_filepath": "/b.wav", "text": "ɛ"}])
    _write_jsonl(corpus / "test.json", [])
    return corpus


def test_current_twi_profile_produces_review_packet_not_fake_alphabet(tmp_path: Path) -> None:
    """Observed corpus characters should be surfaced while current unreviewed policy remains blocked."""
    corpus = _write_tts_artifact(tmp_path)
    report = build_readiness_report(
        profile_path=ROOT / "training/configs/languages/tw.yaml",
        artifacts=corpus,
    )
    codes = {item["code"] for item in report["blockers"]}
    assert {"frontend_unselected", "tokenizer_unreviewed", "graphemes_unreviewed"}.issubset(codes)
    assert report["ready_for_production_training"] is False
    assert report["candidate_inventory_is_approved"] is False
    assert {item["char"] for item in report["grapheme_review_candidates"]} == {"a", "ɛ"}
    assert report["source_summary"][0]["source"] == "waxal_twi_tts"


def test_empty_dataset_version_cannot_be_frozen(tmp_path: Path) -> None:
    """A compiler artifact with zero accepted rows is diagnostic output, not a valid corpus-v0 freeze."""
    artifact = tmp_path / "empty"
    artifact.mkdir()
    (artifact / "dataset_version.json").write_text(
        json.dumps({"accepted": 0, "fingerprint_sha256": "b" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="empty corpus"):
        _read_version(artifact)
