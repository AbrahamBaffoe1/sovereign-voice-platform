"""End-to-end corpus-v0 builder: acquire governed public data, freeze manifests, and prove holdout separation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.data.acquire import dry_run_plan, target_sample_rate
from training.data.bootstrap_plan import BootstrapPlan
from training.data.catalog import SourceCatalog
from training.data.leakage import assert_no_exact_audio_leakage

_TARGET_LANGUAGES = ("tw", "gaa", "ee", "ha")
_TARGET_TASKS = ("asr", "tts")
_GIB = 1024**3


def _now() -> str:
    """Return a reproducible timezone-aware build timestamp."""
    return datetime.now(UTC).isoformat()


def _nearest_existing(path: Path) -> Path:
    """Find the mount backing a not-yet-created output path without creating files during dry-run."""
    current = path.expanduser().resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _disk_snapshot(path: Path) -> dict[str, Any]:
    """Report capacity for the filesystem that will hold a corpus/artifact root."""
    anchor = _nearest_existing(path)
    usage = shutil.disk_usage(anchor)
    return {
        "requested_path": str(path),
        "filesystem_anchor": str(anchor),
        "total_gb": round(usage.total / _GIB, 3),
        "used_gb": round(usage.used / _GIB, 3),
        "free_gb": round(usage.free / _GIB, 3),
    }


def _preflight(*, data_root: Path, artifacts_root: Path, min_free_gb: float) -> dict[str, Any]:
    """Fail before network acquisition when operator-defined persistent-storage headroom is unavailable."""
    if min_free_gb < 0:
        raise ValueError("min_free_gb cannot be negative")
    data = _disk_snapshot(data_root)
    artifacts = _disk_snapshot(artifacts_root)
    if float(data["free_gb"]) < min_free_gb:
        raise RuntimeError(
            f"data filesystem has {data['free_gb']} GiB free; --min-free-gb requires {min_free_gb} GiB"
        )
    if float(artifacts["free_gb"]) < min_free_gb:
        raise RuntimeError(
            f"artifact filesystem has {artifacts['free_gb']} GiB free; --min-free-gb requires {min_free_gb} GiB"
        )
    return {"data": data, "artifacts": artifacts, "minimum_free_gb": min_free_gb}


def _compile(
    *,
    language: str,
    task: str,
    metadata: Path,
    artifacts_root: Path,
    profiles_dir: Path,
    fixed_split: str | None = None,
) -> Path:
    """Run the strict compiler with the same task-specific sample-rate policy used during acquisition."""
    profile = profiles_dir / f"{language}.yaml"
    suffix = "corpus-v0-eval" if fixed_split else "corpus-v0"
    output = artifacts_root / language / task / suffix
    sample_rate = target_sample_rate(language=language, task=task, profiles_dir=profiles_dir)
    command = [
        sys.executable,
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
        str(sample_rate),
    ]
    if fixed_split is not None:
        command.extend(["--fixed-split", fixed_split])
    subprocess.run(command, check=True)
    version_path = output / "dataset_version.json"
    if not version_path.exists():
        raise RuntimeError(f"compiler completed without dataset version: {version_path}")
    return output


def _acquire_in_worker(
    *,
    language: str,
    task: str,
    catalog: Path,
    plan: Path,
    profiles_dir: Path,
    data_root: Path,
    include_eval: bool,
    max_samples: int | None,
    refresh_lock: bool,
    force_reacquire: bool,
) -> dict[str, object]:
    """Run provider libraries in a disposable process, then read their durable summary from disk."""
    command = [
        sys.executable,
        "-m",
        "training.data.acquire_worker",
        "--language",
        language,
        "--task",
        task,
        "--catalog",
        str(catalog),
        "--plan",
        str(plan),
        "--profiles-dir",
        str(profiles_dir),
        "--output-root",
        str(data_root),
    ]
    if include_eval:
        command.append("--include-eval")
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if refresh_lock:
        command.append("--refresh-lock")
    if force_reacquire:
        command.append("--force-reacquire")
    subprocess.run(command, check=True)
    summary_path = data_root / language / task / "ACQUISITION_SUMMARY.json"
    if not summary_path.exists():
        raise RuntimeError(f"acquisition worker exited without summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid acquisition summary: {summary_path}")
    return payload


def _read_version(artifact_dir: Path) -> dict[str, Any]:
    """Return a non-empty immutable compiler identity for the top-level build report."""
    path = artifact_dir / "dataset_version.json"
    if not path.exists():
        raise RuntimeError(f"missing dataset version: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid dataset version: {path}")
    accepted = int(payload.get("accepted", 0))
    fingerprint = str(payload.get("fingerprint_sha256") or "").strip()
    if accepted < 1:
        raise RuntimeError(
            f"refusing to freeze empty corpus artifact: {artifact_dir}; inspect rejected.json and quality_report.json"
        )
    if len(fingerprint) != 64:
        raise RuntimeError(f"dataset version has invalid fingerprint: {path}")
    return payload


def _freeze_one(
    *,
    language: str,
    task: str,
    catalog_path: Path,
    plan: BootstrapPlan,
    plan_path: Path,
    profiles_dir: Path,
    data_root: Path,
    artifacts_root: Path,
    include_eval: bool,
    require_eval: bool,
    max_samples: int | None,
    refresh_lock: bool,
    force_reacquire: bool,
) -> dict[str, Any]:
    """Acquire and freeze one language/task corpus plus an independently compiled held-out benchmark."""
    summary = _acquire_in_worker(
        language=language,
        task=task,
        catalog=catalog_path,
        plan=plan_path,
        profiles_dir=profiles_dir,
        data_root=data_root,
        include_eval=include_eval,
        max_samples=max_samples,
        refresh_lock=refresh_lock,
        force_reacquire=force_reacquire,
    )
    metadata_value = summary.get("train_metadata")
    if not metadata_value:
        raise RuntimeError(f"no training rows acquired for {language}/{task}")
    training_dir = _compile(
        language=language,
        task=task,
        metadata=Path(str(metadata_value)),
        artifacts_root=artifacts_root,
        profiles_dir=profiles_dir,
    )
    training_version = _read_version(training_dir)
    result: dict[str, Any] = {
        "language": language,
        "task": task,
        "sample_rate": target_sample_rate(language=language, task=task, profiles_dir=profiles_dir),
        "acquisition": summary,
        "training_artifact_dir": str(training_dir),
        "training_version": training_version,
        "evaluation_artifact_dir": None,
        "evaluation_version": None,
        "leakage_report": None,
    }

    expected_eval = plan.sources(language=language, task=task, role="eval")
    evaluation_value = summary.get("evaluation_metadata")
    if include_eval and expected_eval and not evaluation_value and require_eval:
        failures = summary.get("optional_failures") or []
        raise RuntimeError(
            f"evaluation was required for {language}/{task} but no evaluation rows were acquired; failures={failures}"
        )
    if evaluation_value:
        evaluation_dir = _compile(
            language=language,
            task=task,
            metadata=Path(str(evaluation_value)),
            artifacts_root=artifacts_root,
            profiles_dir=profiles_dir,
            fixed_split="test",
        )
        evaluation_version = _read_version(evaluation_dir)
        leakage_path = artifacts_root / language / task / "exact_audio_leakage.json"
        leakage = assert_no_exact_audio_leakage(
            training_audit=training_dir / "audit.jsonl",
            evaluation_audit=evaluation_dir / "audit.jsonl",
            report_path=leakage_path,
        )
        result.update(
            {
                "evaluation_artifact_dir": str(evaluation_dir),
                "evaluation_version": evaluation_version,
                "leakage_report": leakage,
            }
        )
    return result


def main() -> None:
    """Build public corpus-v0 for one or all target languages with durable provenance and holdout proofs."""
    parser = argparse.ArgumentParser(description="Acquire, validate, and freeze governed public speech corpus-v0")
    parser.add_argument("--language", choices=[*_TARGET_LANGUAGES, "all"], default="all")
    parser.add_argument("--task", choices=[*_TARGET_TASKS, "both"], default="both")
    parser.add_argument("--catalog", type=Path, default=Path("training/configs/source_catalog.yaml"))
    parser.add_argument("--plan", type=Path, default=Path("training/configs/bootstrap_corpora.yaml"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("training/configs/languages"))
    parser.add_argument("--data-root", type=Path, default=Path("data/bootstrap"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/bootstrap"))
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refresh-lock", action="store_true")
    parser.add_argument("--force-reacquire", action="store_true")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=0.0,
        help="Operator-defined minimum free space required on both data and artifact filesystems.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.require_eval:
        args.include_eval = True
    languages = list(_TARGET_LANGUAGES) if args.language == "all" else [args.language]
    tasks = list(_TARGET_TASKS) if args.task == "both" else [args.task]
    catalog = SourceCatalog(args.catalog)
    plan = BootstrapPlan(args.plan)
    preflight = _preflight(
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        min_free_gb=args.min_free_gb,
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "started_at": _now(),
        "status": "planning" if args.dry_run else "in_progress",
        "data_root": str(args.data_root),
        "artifacts_root": str(args.artifacts_root),
        "include_eval": args.include_eval,
        "require_eval": args.require_eval,
        "max_samples": args.max_samples,
        "preflight": preflight,
        "runs": [],
    }

    for language in languages:
        for task in tasks:
            if args.dry_run:
                report["runs"].append(
                    {
                        "language": language,
                        "task": task,
                        "sample_rate": target_sample_rate(
                            language=language,
                            task=task,
                            profiles_dir=args.profiles_dir,
                        ),
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
            report["runs"].append(
                _freeze_one(
                    language=language,
                    task=task,
                    catalog_path=args.catalog,
                    plan=plan,
                    plan_path=args.plan,
                    profiles_dir=args.profiles_dir,
                    data_root=args.data_root,
                    artifacts_root=args.artifacts_root,
                    include_eval=args.include_eval,
                    require_eval=args.require_eval,
                    max_samples=args.max_samples,
                    refresh_lock=args.refresh_lock,
                    force_reacquire=args.force_reacquire,
                )
            )

    report["completed_at"] = _now()
    report["status"] = "planned" if args.dry_run else "completed"
    report["postflight"] = {
        "data": _disk_snapshot(args.data_root),
        "artifacts": _disk_snapshot(args.artifacts_root),
    }
    if not args.dry_run:
        args.artifacts_root.mkdir(parents=True, exist_ok=True)
        report_path = args.artifacts_root / "BUILD_REPORT.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["build_report"] = str(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
