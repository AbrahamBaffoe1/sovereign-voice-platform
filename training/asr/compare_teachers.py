"""Compare aligned ASR teacher predictions and evaluate a confidence-selected ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.asr.teacher_benchmark import _whisper_basic_summary
from training.benchmarks.speech import BenchmarkUtterance, benchmark_report, normalize_for_error_rate
from training.common.manifest import file_sha256, read_jsonl


def _aligned_rows(left_path: Path, right_path: Path) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Load two prediction streams and reject any benchmark row drift before comparison."""
    left = read_jsonl(left_path)
    right = read_jsonl(right_path)
    if len(left) != len(right):
        raise ValueError(f"prediction row-count mismatch: {len(left)} != {len(right)}")
    aligned: list[tuple[dict[str, object], dict[str, object]]] = []
    for index, (a, b) in enumerate(zip(left, right, strict=True), 1):
        keys = ("audio_filepath", "reference")
        if any(str(a.get(key)) != str(b.get(key)) for key in keys):
            raise ValueError(f"prediction alignment mismatch at row {index}")
        aligned.append((a, b))
    return aligned


def compare_predictions(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Report each teacher plus exact-agreement and confidence-selection ensemble behavior."""
    pairs = _aligned_rows(left_path, right_path)
    left_rows: list[BenchmarkUtterance] = []
    right_rows: list[BenchmarkUtterance] = []
    ensemble_rows: list[BenchmarkUtterance] = []
    agreed = 0
    left_selected = 0
    right_selected = 0
    for left, right in pairs:
        reference = str(left["reference"])
        left_hyp = str(left["hypothesis"])
        right_hyp = str(right["hypothesis"])
        audio_seconds = float(left.get("audio_seconds") or 0.0)
        left_latency = float(left.get("latency_seconds") or 0.0)
        right_latency = float(right.get("latency_seconds") or 0.0)
        left_conf = float(left.get("teacher_confidence") or 0.0)
        right_conf = float(right.get("teacher_confidence") or 0.0)
        shared = dict(
            reference=reference,
            audio_seconds=audio_seconds,
            speaker=str(left["speaker"]) if left.get("speaker") else None,
            dialect=str(left["dialect"]) if left.get("dialect") else None,
            noise=str(left["noise"]) if left.get("noise") else None,
            code_switch=str(left["code_switch"]) if left.get("code_switch") else None,
        )
        left_rows.append(BenchmarkUtterance(hypothesis=left_hyp, latency_seconds=left_latency, **shared))
        right_rows.append(BenchmarkUtterance(hypothesis=right_hyp, latency_seconds=right_latency, **shared))
        agreed += int(normalize_for_error_rate(left_hyp) == normalize_for_error_rate(right_hyp))
        if left_conf >= right_conf:
            selected_hyp, selected_latency = left_hyp, left_latency
            left_selected += 1
        else:
            selected_hyp, selected_latency = right_hyp, right_latency
            right_selected += 1
        ensemble_rows.append(BenchmarkUtterance(hypothesis=selected_hyp, latency_seconds=selected_latency, **shared))

    def metrics(rows: list[BenchmarkUtterance]) -> dict[str, Any]:
        report = benchmark_report(rows)
        try:
            report["whisper_basic"] = _whisper_basic_summary(rows)
        except ModuleNotFoundError:
            report["whisper_basic"] = {"available": False, "reason": "transformers_not_installed"}
        return report

    return {
        "schema_version": 1,
        "rows": len(pairs),
        "left": {"path": str(left_path), "sha256": file_sha256(left_path), "metrics": metrics(left_rows)},
        "right": {"path": str(right_path), "sha256": file_sha256(right_path), "metrics": metrics(right_rows)},
        "agreement": {"utterances": agreed, "rate": agreed / len(pairs) if pairs else 0.0},
        "confidence_selected": {
            "left_selected": left_selected,
            "right_selected": right_selected,
            "metrics": metrics(ensemble_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two aligned ASR teacher prediction files")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = compare_predictions(args.left, args.right)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
