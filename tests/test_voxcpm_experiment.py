"""VoxCPM2 adaptation-plan tests."""

import json
from pathlib import Path

from training.tts.build_voxcpm_experiment import build_experiment


def test_voxcpm_plan_is_bound_to_dataset_fingerprint(tmp_path: Path) -> None:
    """A TTS candidate should retain exact data lineage before vendor trainer execution."""
    version = tmp_path / "dataset_version.json"
    version.write_text(
        json.dumps({"dataset_id": "tw-1", "fingerprint_sha256": "a" * 64}), encoding="utf-8"
    )
    plan = build_experiment(
        language="tw",
        dataset_version=version,
        mode="lora",
        output_root=tmp_path / "artifacts",
    )
    assert plan.dataset_fingerprint == "a" * 64
    assert plan.experiment_id.startswith("tw-voxcpm2-lora-")
