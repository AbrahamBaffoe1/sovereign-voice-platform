"""Frozen experiment-plan execution tests."""

import json
from pathlib import Path

import pytest

from training.experiments.plans import build_plan
from training.experiments.runner import command_for, render_kubernetes


def test_whisper_plan_is_deterministic_and_renderable(tmp_path: Path) -> None:
    """Cluster rendering must preserve the exact frozen scientific plan identity."""
    version = tmp_path / "dataset_version.json"
    version.write_text(
        json.dumps({"dataset_id": "tw-abc", "fingerprint_sha256": "f" * 64}), encoding="utf-8"
    )
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        "candidates:\n  whisper-small:\n    family: whisper\n    base_model: openai/whisper-small\n    trainable: true\n",
        encoding="utf-8",
    )
    plan = build_plan(
        language="tw",
        dataset_version=version,
        candidate="whisper-small",
        candidates_config=candidates,
        artifacts_dir=tmp_path / "artifacts",
    )
    assert command_for(plan)[0:3] == ["python", "-m", "training.asr.finetune_whisper"]
    rendered = json.loads(render_kubernetes(plan, image="trainer:test"))
    assert rendered["metadata"]["name"] == plan.experiment_id


def test_unimplemented_candidate_fails_closed(tmp_path: Path) -> None:
    """Visible research candidates must not generate a fake training command."""
    version = tmp_path / "dataset_version.json"
    version.write_text(
        json.dumps({"dataset_id": "tw-abc", "fingerprint_sha256": "f" * 64}), encoding="utf-8"
    )
    candidates = tmp_path / "candidates.yaml"
    candidates.write_text(
        "candidates:\n  alt:\n    family: w2v-bert\n    trainable: false\n    blocked_reason: tokenizer missing\n",
        encoding="utf-8",
    )
    plan = build_plan(
        language="tw",
        dataset_version=version,
        candidate="alt",
        candidates_config=candidates,
        artifacts_dir=tmp_path / "artifacts",
    )
    with pytest.raises(RuntimeError, match="tokenizer missing"):
        command_for(plan)
