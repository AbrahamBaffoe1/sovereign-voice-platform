"""CLI for aggregating normalized ASR benchmark JSONL into a metrics artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.benchmarks.speech import benchmark_report, rows_from_jsonl


def main() -> None:
    """Compute the speech benchmark report and optionally persist it for model registration."""
    parser = argparse.ArgumentParser(description="Aggregate ASR benchmark rows")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = rows_from_jsonl(args.input)
    if not rows:
        raise SystemExit("benchmark input contains no utterances")
    report = benchmark_report(rows)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
