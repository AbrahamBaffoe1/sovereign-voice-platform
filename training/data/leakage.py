"""Cross-corpus leakage checks for frozen training and evaluation speech artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _audit_hashes(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Index audit rows by normalized audio SHA-256 with precise input validation."""
    if not path.exists():
        raise FileNotFoundError(path)
    indexed: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            digest = str(payload.get("sha256") or "").strip().lower()
            if len(digest) != 64:
                raise ValueError(f"{path}:{line_number}: missing or invalid sha256")
            indexed.setdefault(digest, []).append(payload)
    return indexed


def exact_audio_leakage_report(*, training_audit: Path, evaluation_audit: Path) -> dict[str, Any]:
    """Return every normalized waveform that appears in both frozen train and evaluation corpora."""
    training = _audit_hashes(training_audit)
    evaluation = _audit_hashes(evaluation_audit)
    overlap = sorted(training.keys() & evaluation.keys())
    examples: list[dict[str, Any]] = []
    for digest in overlap[:100]:
        examples.append(
            {
                "sha256": digest,
                "training_source_ids": sorted(
                    {str(row.get("source_id") or "") for row in training[digest] if row.get("source_id")}
                ),
                "evaluation_source_ids": sorted(
                    {str(row.get("source_id") or "") for row in evaluation[digest] if row.get("source_id")}
                ),
                "training_audio": sorted(
                    {str(row.get("audio_filepath") or "") for row in training[digest] if row.get("audio_filepath")}
                ),
                "evaluation_audio": sorted(
                    {str(row.get("audio_filepath") or "") for row in evaluation[digest] if row.get("audio_filepath")}
                ),
            }
        )
    return {
        "schema_version": 1,
        "training_rows": sum(len(rows) for rows in training.values()),
        "evaluation_rows": sum(len(rows) for rows in evaluation.values()),
        "overlap_count": len(overlap),
        "passed": not overlap,
        "overlap_examples": examples,
    }


def assert_no_exact_audio_leakage(
    *,
    training_audit: Path,
    evaluation_audit: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Persist the leakage report and fail closed when a waveform crosses the holdout boundary."""
    report = exact_audio_leakage_report(
        training_audit=training_audit,
        evaluation_audit=evaluation_audit,
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(
            f"exact audio leakage detected between training and evaluation: {report['overlap_count']} waveform(s)"
        )
    return report
