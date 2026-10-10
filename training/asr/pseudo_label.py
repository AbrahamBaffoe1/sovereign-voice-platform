"""Generate TRAIN-only pseudo labels from a registered public CTC teacher with durable provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.asr.teacher_benchmark import _ctc_transcribe, _load_audio, _load_ctc_teacher
from training.asr.teacher_registry import get_teacher
from training.common.manifest import file_sha256, read_jsonl


def _training_rows(path: Path, *, limit: int | None) -> list[dict[str, object]]:
    rows = read_jsonl(path)
    selected: list[dict[str, object]] = []
    for row in rows:
        split = row.get("split") or row.get("source_split")
        if split not in {None, "train"}:
            raise ValueError(f"refusing pseudo-label input containing non-train split: {split!r}")
        operational_class = row.get("operational_class")
        if operational_class not in {None, "TRAIN"}:
            raise ValueError(
                f"refusing pseudo-label input with non-training operational class: {operational_class!r}"
            )
        if row.get("allow_optimizer_updates") is False:
            raise ValueError("refusing pseudo-label input explicitly barred from optimizer updates")
        audio_path = row.get("audio_filepath") or row.get("audio_path")
        if not audio_path:
            raise ValueError(f"{path}: row missing audio path")
        normalized = dict(row)
        normalized["audio_filepath"] = str(audio_path)
        selected.append(normalized)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"no training rows selected from {path}")
    return selected


def generate_pseudo_labels(
    *,
    teacher_name: str,
    manifest: Path,
    output: Path,
    limit: int | None = None,
    device: str = "auto",
    min_confidence: float = 0.0,
    model_source: str | None = None,
) -> dict[str, object]:
    """Infer labels only for training examples; validation/test manifests are rejected by construction."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    teacher = get_teacher(teacher_name)
    if not teacher.runnable:
        raise RuntimeError(teacher.blocked_reason or f"teacher {teacher_name} is not runnable")
    if teacher.family != "ctc":
        raise RuntimeError(f"teacher family {teacher.family!r} is not implemented yet")
    rows = _training_rows(manifest, limit=limit)
    source_manifest_sha256 = file_sha256(manifest)
    resolved_model_source = model_source or teacher.model_id
    processor, model, torch = _load_ctc_teacher(resolved_model_source, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            audio_path = Path(str(row["audio_filepath"]))
            waveform, duration = _load_audio(audio_path)
            hypothesis, confidence = _ctc_transcribe(
                processor, model, torch, waveform, language_prefix_id=teacher.language_prefix_id
            )
            hypothesis = hypothesis.strip()
            if not hypothesis or confidence < min_confidence:
                continue
            payload: dict[str, Any] = {
                "audio_filepath": str(audio_path),
                "text": hypothesis,
                "duration": float(row.get("duration") or duration),
                "speaker": row.get("speaker"),
                "language": row.get("language") or teacher.language,
                "dialect": row.get("dialect"),
                "source_id": row.get("source_id"),
                "split": "train",
                "label_origin": "teacher_pseudo_label",
                "teacher_name": teacher.name,
                "teacher_model_id": teacher.model_id,
                "teacher_family": teacher.family,
                "teacher_language_prefix_id": teacher.language_prefix_id,
                "teacher_model_source": resolved_model_source,
                "teacher_confidence": confidence,
                "source_manifest": str(manifest),
                "source_manifest_sha256": source_manifest_sha256,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            emitted += 1
            print(f"[{index}/{len(rows)}] {audio_path.name} confidence={confidence:.4f}", flush=True)
    lineage = {
        "schema_version": 1,
        "teacher": teacher.name,
        "model_id": teacher.model_id,
        "model_source": resolved_model_source,
        "language_prefix_id": teacher.language_prefix_id,
        "input_manifest": str(manifest),
        "input_manifest_sha256": source_manifest_sha256,
        "input_rows": len(rows),
        "emitted_rows": emitted,
        "output_manifest": str(output),
        "output_manifest_sha256": file_sha256(output),
        "split_policy": "train_only",
        "min_confidence": min_confidence,
    }
    output.with_suffix(output.suffix + ".lineage.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return lineage


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe teacher pseudo labels")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--model-source", default=None, help="Optional local checkpoint path")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    print(
        json.dumps(
            generate_pseudo_labels(
                teacher_name=args.teacher,
                manifest=args.manifest,
                output=args.output,
                limit=args.limit,
                device=args.device,
                min_confidence=args.min_confidence,
                model_source=args.model_source,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
