"""End-to-end bootstrap command: acquire approved public data, compile it, and freeze corpus-v0 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from training.data.acquire import acquire_language, dry_run_plan
from training.data.bootstrap_plan import BootstrapPlan
from training.data.catalog import SourceCatalog


def _compile(
    *,
    language: str,
    task: str,
    metadata: Path,
    artifacts_root: Path,
) -> Path:
    """Invoke the same strict corpus compiler used for first-party data; only governance inputs differ."""
    profile = Path("training/configs/languages") / f"{language}.yaml"
    output = artifacts_root / language / task / "corpus-v0"
    command = [
        "python",
        "-m",
        "training.prepare_dataset",
        "--profile",
        str(profile),
        "--csv",
        str(metadata),
        "--audio-root",
        ".",
        "--output",
        str(output),
        "--required-sample-rate",
        "16000",
    ]
    subprocess.run(command, check=True)
    return output


def main() -> None:
    """Build public corpus-v0 for one or all target languages while keeping evaluation data separate."""
    parser = argparse.ArgumentParser(description="Acquire and compile bootstrap speech corpus-v0")
    parser.add_argument("--language", choices=["tw", "gaa", "ee", "ha", "all"], default="all")
    parser.add_argument("--task", choices=["asr", "tts", "both"], default="both")
    parser.add_argument("--catalog", type=Path, default=Path("training/configs/source_catalog.yaml"))
    parser.add_argument("--plan", type=Path, default=Path("training/configs/bootstrap_corpora.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/bootstrap"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/bootstrap"))
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refresh-lock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    languages = ["tw", "gaa", "ee", "ha"] if args.language == "all" else [args.language]
    tasks = ["asr", "tts"] if args.task == "both" else [args.task]
    catalog = SourceCatalog(args.catalog)
    plan = BootstrapPlan(args.plan)
    report: dict[str, object] = {"runs": []}
    for language in languages:
        for task in tasks:
            if args.dry_run:
                report["runs"].append(
                    {
                        "language": language,
                        "task": task,
                        "plan": dry_run_plan(
                            language=language,
                            task=task,
                            catalog=catalog,
                            plan=plan,
                            include_eval=args.include_eval,
                        ),
                    }
                )
                continue
            summary = acquire_language(
                language=language,
                task=task,
                catalog_path=args.catalog,
                plan_path=args.plan,
                output_root=args.data_root,
                include_eval=args.include_eval,
                max_samples=args.max_samples,
                refresh_lock=args.refresh_lock,
                token=os.environ.get("HF_TOKEN") or None,
            )
            metadata_value = summary.get("train_metadata")
            if not metadata_value:
                raise RuntimeError(f"no training rows acquired for {language}/{task}")
            artifact_dir = _compile(
                language=language,
                task=task,
                metadata=Path(str(metadata_value)),
                artifacts_root=args.artifacts_root,
            )
            report["runs"].append(
                {
                    "language": language,
                    "task": task,
                    "acquisition": summary,
                    "artifact_dir": str(artifact_dir),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
