"""Tests for lineage-bound Whisper baseline planning without loading model frameworks."""

from __future__ import annotations

import json
from pathlib import Path

from training.asr.run_baseline import build_language_plan

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _freeze_stub(root: Path, *, language: str, validation: bool, external: bool) -> None:
    corpus = root / language / "asr" / "corpus-v0"
    corpus.mkdir(parents=True, exist_ok=True)
    version = {
        "accepted": 3,
        "fingerprint_sha256": "a" * 64,
        "dataset_id": f"{language}-stub",
    }
    (corpus / "dataset_version.json").write_text(json.dumps(version), encoding="utf-8")
    _write_jsonl(corpus / "train.json", [{"audio_filepath": "/a.wav", "text": "one"}])
    _write_jsonl(
        corpus / "validation.json",
        [{"audio_filepath": "/b.wav", "text": "two"}] if validation else [],
    )
    _write_jsonl(corpus / "test.json", [])
    _write_jsonl(
        corpus / "audit.jsonl",
        [
            {
                "audio_filepath": "/a.wav",
                "text": "one",
                "sha256": "1" * 64,
                "split": "train",
            }
        ],
    )
    if external:
        evaluation = root / language / "asr" / "corpus-v0-eval"
        evaluation.mkdir(parents=True, exist_ok=True)
        (evaluation / "dataset_version.json").write_text(
            json.dumps({"accepted": 1, "fingerprint_sha256": "b" * 64}),
            encoding="utf-8",
        )
        _write_jsonl(
            evaluation / "audit.jsonl",
            [
                {
                    "audio_filepath": "/e.wav",
                    "text": "held out",
                    "sha256": "2" * 64,
                    "split": "test",
                }
            ],
        )


def _plan(
    tmp_path: Path,
    *,
    validation: bool,
    external: bool,
    require_external: bool = False,
    resume: bool = False,
) -> dict:
    artifacts = tmp_path / "bootstrap"
    _freeze_stub(artifacts, language="tw", validation=validation, external=external)
    return build_language_plan(
        language="tw",
        artifacts_root=artifacts,
        profiles_dir=ROOT / "training/configs/languages",
        output_root=tmp_path / "experiments",
        max_steps=10,
        batch_size=2,
        gradient_accumulation=1,
        learning_rate=1e-5,
        precision="fp32",
        quantization="float32",
        require_external_eval=require_external,
        device="cpu",
        compute_type="float32",
        resume=resume,
    )


def test_baseline_without_internal_validation_uses_fixed_steps(tmp_path: Path) -> None:
    """Speaker-unknown training-only corpora must not borrow the external benchmark for checkpoint selection."""
    plan = _plan(tmp_path, validation=False, external=True, require_external=True)
    assert plan["selection_policy"] == "fixed_steps_final"
    assert "--validation" not in plan["train_command"]
    assert [item["name"] for item in plan["evaluations"]] == ["external_test"]


def test_baseline_with_internal_validation_selects_by_wer(tmp_path: Path) -> None:
    """A genuine internal validation split can select checkpoints while the external benchmark stays post-training."""
    plan = _plan(tmp_path, validation=True, external=True, require_external=True)
    assert plan["selection_policy"] == "best_validation_wer"
    assert "--validation" in plan["train_command"]
    validation_path = str(tmp_path / "bootstrap/tw/asr/corpus-v0/validation.json")
    assert validation_path in plan["train_command"]
    assert all("corpus-v0-eval" not in str(part) for part in plan["train_command"])
    assert [item["name"] for item in plan["evaluations"]] == ["external_test"]


def test_resume_plan_requests_auto_checkpoint_without_changing_lineage(tmp_path: Path) -> None:
    """The production runner may always request resume; a fresh run resolves auto to no checkpoint."""
    plan = _plan(tmp_path, validation=True, external=False, resume=True)
    command = plan["train_command"]
    position = command.index("--resume-from-checkpoint")
    assert command[position + 1] == "auto"
    assert plan["resume"] is True
