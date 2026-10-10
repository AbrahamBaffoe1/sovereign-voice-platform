"""Join accepted teacher labels back to governed TRAIN sources for the Nexora LoRA trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common.manifest import file_sha256, read_jsonl

_SCHEMA = "nexora.distillation-row.v1"


def _eval_hashes(path: Path) -> set[str]:
    """Read frozen evaluation hashes so no held-out audio can enter optimizer input."""
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if len(token) == 64:
            hashes.add(token)
    return hashes


def build_distillation_manifest(
    *,
    source_manifest: Path,
    pseudo_manifest: Path,
    output: Path,
    frozen_eval_hashes: Path,
) -> dict[str, Any]:
    """Produce trainer-native pseudo rows only when source governance and leakage checks pass."""
    source_rows = read_jsonl(source_manifest)
    pseudo_rows = read_jsonl(pseudo_manifest)
    held_out = _eval_hashes(frozen_eval_hashes)
    by_audio: dict[str, dict[str, object]] = {}
    for row in source_rows:
        audio_path = str(row.get("audio_path") or row.get("audio_filepath") or "")
        if not audio_path:
            raise ValueError("source row missing audio path")
        if audio_path in by_audio:
            raise ValueError(f"duplicate source audio path: {audio_path}")
        by_audio[audio_path] = row

    seen_hashes: set[str] = set()
    rendered: list[dict[str, Any]] = []
    for pseudo in pseudo_rows:
        audio_path = str(pseudo.get("audio_filepath") or pseudo.get("audio_path") or "")
        source = by_audio.get(audio_path)
        if source is None:
            raise ValueError(f"pseudo label has no governed source row: {audio_path}")
        if source.get("operational_class") != "TRAIN" or source.get("allow_optimizer_updates") is not True:
            raise ValueError(f"source is not optimizer-authorized TRAIN data: {audio_path}")
        if source.get("commercial_training_eligible") is not True:
            raise ValueError(f"source is not commercially training eligible: {audio_path}")
        audio_hash = str(source.get("audio_sha256") or "")
        if len(audio_hash) != 64:
            raise ValueError(f"source row missing valid audio_sha256: {audio_path}")
        if audio_hash in held_out:
            raise ValueError(f"frozen evaluation leakage detected: {audio_path}")
        if audio_hash in seen_hashes:
            raise ValueError(f"duplicate pseudo-label audio hash: {audio_hash}")
        text = str(pseudo.get("text") or pseudo.get("target_text") or "").strip()
        if not text:
            raise ValueError(f"empty pseudo target: {audio_path}")
        confidence = float(pseudo.get("teacher_confidence") or 0.0)
        rendered.append(
            {
                "schema_version": _SCHEMA,
                "operational_class": "TRAIN",
                "allow_optimizer_updates": True,
                "commercial_training_eligible": True,
                "audio_path": audio_path,
                "audio_sha256": audio_hash,
                "target_text": text,
                "target_source": "teacher-pseudo-label",
                "human_text": None,
                "teacher_targets": [],
                "teacher_name": pseudo.get("teacher_name"),
                "teacher_model_id": pseudo.get("teacher_model_id"),
                "teacher_confidence": confidence,
                "source_id": source.get("source_id"),
                "sample_id": source.get("sample_id"),
                "speaker_id": source.get("speaker_id"),
                "language": source.get("language"),
                "domains": source.get("domains") or [],
                "source_repository": source.get("source_repository"),
                "source_revision": source.get("source_revision"),
                "source_license": source.get("license"),
                "source_manifest_sha256": file_sha256(source_manifest),
                "pseudo_manifest_sha256": file_sha256(pseudo_manifest),
            }
        )
        seen_hashes.add(audio_hash)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rendered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lineage = {
        "schema_version": 1,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "pseudo_manifest": str(pseudo_manifest),
        "pseudo_manifest_sha256": file_sha256(pseudo_manifest),
        "frozen_eval_hashes": str(frozen_eval_hashes),
        "frozen_eval_hashes_sha256": file_sha256(frozen_eval_hashes),
        "source_rows": len(source_rows),
        "pseudo_rows": len(pseudo_rows),
        "output_rows": len(rendered),
        "output_manifest": str(output),
        "output_manifest_sha256": file_sha256(output),
        "policy": {
            "operational_class": "TRAIN",
            "allow_optimizer_updates": True,
            "commercial_training_eligible": True,
            "frozen_eval_hashes_forbidden": True,
            "target_source": "teacher-pseudo-label",
        },
    }
    output.with_suffix(output.suffix + ".lineage.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return lineage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nexora trainer-native pseudo-label manifest")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pseudo-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-eval-hashes", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_distillation_manifest(
                source_manifest=args.source_manifest,
                pseudo_manifest=args.pseudo_manifest,
                output=args.output,
                frozen_eval_hashes=args.frozen_eval_hashes,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
