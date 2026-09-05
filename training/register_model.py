"""Register a trained checkpoint as a traceable candidate model with dataset and metric lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common.language_profile import load_language_profile
from training.common.model_card import ModelCard, build_model_id, hash_checkpoint_tree, utc_now


def parse_args() -> argparse.Namespace:
    """Collect immutable lineage inputs required before a checkpoint may enter model review."""
    parser = argparse.ArgumentParser(description="Create model_card.json for a trained checkpoint")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=("asr", "tts-acoustic", "tts-vocoder"), required=True)
    parser.add_argument("--dataset-version", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--base-model", default=None)
    return parser.parse_args()


def main() -> None:
    """Validate lineage artifacts, hash checkpoint contents and write a candidate model card."""
    args = parse_args()
    profile = load_language_profile(args.profile)
    dataset = json.loads(args.dataset_version.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if dataset.get("language") != profile.code:
        raise SystemExit("dataset version language does not match model language profile")
    fingerprint = str(dataset.get("fingerprint_sha256", ""))
    dataset_id = str(dataset.get("dataset_id", ""))
    if not fingerprint or not dataset_id:
        raise SystemExit("dataset_version.json is missing dataset_id/fingerprint_sha256")
    artifact_hash, artifact_files = hash_checkpoint_tree(args.checkpoint)
    card = ModelCard(
        schema_version=1,
        model_id=build_model_id(profile.code, args.task, artifact_hash),
        language=profile.code,
        task=args.task,
        status="candidate",
        created_at=utc_now(),
        dataset_id=dataset_id,
        dataset_fingerprint_sha256=fingerprint,
        base_model=args.base_model,
        checkpoint_path=str(args.checkpoint),
        artifact_manifest_sha256=artifact_hash,
        metrics=metrics,
    )
    (args.checkpoint / "model_card.json").write_text(card.to_json(), encoding="utf-8")
    (args.checkpoint / "artifact_manifest.json").write_text(
        json.dumps(artifact_files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(card.to_json())


if __name__ == "__main__":
    main()
