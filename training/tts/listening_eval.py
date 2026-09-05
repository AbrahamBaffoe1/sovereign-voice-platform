"""Blind native-speaker TTS listening evaluation with deterministic assignment and MOS aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def assignment_bucket(sample_id: str, reviewer: str, buckets: int = 5) -> int:
    """Assign sample/reviewer pairs deterministically so repeated exports keep the same workload."""
    if buckets < 1:
        raise ValueError("buckets must be positive")
    digest = hashlib.sha256(f"{sample_id}\x1f{reviewer}".encode()).hexdigest()
    return int(digest[:8], 16) % buckets


def aggregate(path: Path) -> dict[str, object]:
    """Aggregate 1-5 native ratings by model while preserving rating counts."""
    metrics = ("naturalness", "pronunciation", "intelligibility", "speaker_similarity")
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model_id", "reviewer", "sample_id", *metrics}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ratings CSV missing columns: {sorted(missing)}")
        for row in reader:
            model_id = str(row["model_id"]).strip()
            if not model_id:
                continue
            for metric in metrics:
                value = float(row[metric])
                if not 1.0 <= value <= 5.0:
                    raise ValueError(f"{metric} must be between 1 and 5")
                groups[model_id][metric].append(value)
    return {
        model_id: {
            "ratings": max((len(values) for values in metric_values.values()), default=0),
            **{
                metric: round(statistics.fmean(values), 4) if values else None
                for metric, values in metric_values.items()
            },
        }
        for model_id, metric_values in sorted(groups.items())
    }


def main() -> None:
    """Aggregate a completed blind-listening CSV into a model-comparison JSON artifact."""
    parser = argparse.ArgumentParser(description="Aggregate native-speaker TTS listening ratings")
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rendered = json.dumps(aggregate(args.ratings), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
