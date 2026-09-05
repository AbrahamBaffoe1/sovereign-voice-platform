"""CLI for freezing one ASR candidate/dataset combination into an immutable experiment plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.experiments.plans import build_plan, write_plan


def main() -> None:
    """Build and persist a deterministic ASR plan before cluster execution is requested."""
    parser = argparse.ArgumentParser(description="Create a reproducible ASR experiment plan")
    parser.add_argument("--language", required=True)
    parser.add_argument("--dataset-version", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidates", type=Path, default=Path("training/configs/asr_candidates.yaml"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        language=args.language,
        dataset_version=args.dataset_version,
        candidate=args.candidate,
        candidates_config=args.candidates,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
    )
    write_plan(plan, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
