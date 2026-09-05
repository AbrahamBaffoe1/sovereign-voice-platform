"""Launch reproducible corpus-v0 Whisper baselines without contaminating external benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.common.language_profile import load_language_profile
from training.common.manifest import file_sha256

_LANGUAGES = ("tw", "gaa", "ee", "ha")


def _now() -> str:
    """Return an offset-aware timestamp for experiment lineage."""
    return datetime.now(UTC).isoformat()


def _jsonl_rows(path: Path) -> int:
    """Count manifest rows as a stream so large corpora do not need to fit in memory."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _audit_split_rows(path: Path, split: str) -> int:
    """Count one audit split while validating each JSON object encountered."""
    if not path.exists():
        return 0
    count = 0
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
            if payload.get("split") == split:
                count += 1
    return count


def _read_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object and fail on missing/malformed lineage files."""
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_model_token(model: str) -> str:
    """Convert a model id into a stable path component without losing its human-readable identity."""
    return model.rsplit("/", 1)[-1].replace("_", "-")


def build_language_plan(
    *,
    language: str,
    artifacts_root: Path,
    profiles_dir: Path,
    output_root: Path,
    max_steps: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    precision: str,
    quantization: str,
    require_external_eval: bool,
    device: str,
    compute_type: str,
    resume: bool = False,
) -> dict[str, Any]:
    """Resolve one baseline entirely from frozen corpus artifacts and version-controlled language policy."""
    if language not in _LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if max_steps < 1 or batch_size < 1 or gradient_accumulation < 1 or learning_rate <= 0:
        raise ValueError("training hyperparameters must be positive")
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"unsupported precision: {precision}")

    profile_path = profiles_dir / f"{language}.yaml"
    profile = load_language_profile(profile_path)
    if profile.code != language:
        raise ValueError(f"profile language mismatch: {profile_path}")
    if profile.asr.sample_rate != 16000:
        raise ValueError(f"Whisper baseline requires 16 kHz ASR corpus, got {profile.asr.sample_rate}")

    corpus = artifacts_root / language / "asr" / "corpus-v0"
    version_path = corpus / "dataset_version.json"
    version = _read_json_object(version_path)
    accepted = int(version.get("accepted", 0))
    fingerprint = str(version.get("fingerprint_sha256") or "").strip()
    if accepted < 1 or len(fingerprint) != 64:
        raise ValueError(f"invalid or empty frozen corpus: {version_path}")

    train = corpus / "train.json"
    validation = corpus / "validation.json"
    audit = corpus / "audit.jsonl"
    train_rows = _jsonl_rows(train)
    validation_rows = _jsonl_rows(validation)
    internal_test_rows = _audit_split_rows(audit, "test")
    if train_rows < 1:
        raise ValueError(f"frozen corpus has no training rows: {train}")

    external_corpus = artifacts_root / language / "asr" / "corpus-v0-eval"
    external_version_path = external_corpus / "dataset_version.json"
    external_audit = external_corpus / "audit.jsonl"
    external_rows = _audit_split_rows(external_audit, "test")
    external_version = _read_json_object(external_version_path) if external_version_path.exists() else None
    if external_version is not None and int(external_version.get("accepted", 0)) != external_rows:
        raise ValueError(
            f"external evaluation version/audit row mismatch for {language}: "
            f"accepted={external_version.get('accepted')} test_rows={external_rows}"
        )
    if require_external_eval and external_rows < 1:
        raise ValueError(f"external evaluation is required but unavailable for {language}: {external_corpus}")

    run_name = f"{language}-{_safe_model_token(profile.asr.base_model)}-{fingerprint[:12]}"
    run_dir = output_root / run_name
    hf_dir = run_dir / "hf"
    final_hf = hf_dir / "final"
    ct2_dir = run_dir / "ct2"

    train_command = [
        sys.executable,
        "-m",
        "training.asr.finetune_whisper",
        "--train",
        str(train),
        "--dataset-version",
        str(version_path),
        "--output",
        str(hf_dir),
        "--profile",
        str(profile_path),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--learning-rate",
        str(learning_rate),
    ]
    selection_policy = "fixed_steps_final"
    if validation_rows:
        train_command.extend(["--validation", str(validation)])
        selection_policy = "best_validation_wer"
    if precision == "fp16":
        train_command.append("--fp16")
    elif precision == "bf16":
        train_command.append("--bf16")
    if resume:
        train_command.extend(["--resume-from-checkpoint", "auto"])

    export_command = [
        sys.executable,
        "-m",
        "training.asr.export_ct2",
        "--model",
        str(final_hf),
        "--output",
        str(ct2_dir),
        "--quantization",
        quantization,
    ]
    evaluations: list[dict[str, Any]] = []
    if internal_test_rows:
        evaluations.append(
            {
                "name": "internal_test",
                "rows": internal_test_rows,
                "command": [
                    sys.executable,
                    "-m",
                    "training.asr.evaluate_faster_whisper",
                    "--model",
                    str(ct2_dir),
                    "--manifest",
                    str(audit),
                    "--device",
                    device,
                    "--compute-type",
                    compute_type,
                    "--output",
                    str(run_dir / "internal_test.json"),
                ],
            }
        )
    if external_rows:
        evaluations.append(
            {
                "name": "external_test",
                "rows": external_rows,
                "command": [
                    sys.executable,
                    "-m",
                    "training.asr.evaluate_faster_whisper",
                    "--model",
                    str(ct2_dir),
                    "--manifest",
                    str(external_audit),
                    "--device",
                    device,
                    "--compute-type",
                    compute_type,
                    "--output",
                    str(run_dir / "external_test.json"),
                ],
            }
        )

    return {
        "schema_version": 2,
        "language": language,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "profile": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "base_model": profile.asr.base_model,
        "language_token_mode": profile.asr.language_token_mode,
        "decoder_language": profile.asr.decoder_language,
        "dataset": {
            "artifact_dir": str(corpus),
            "version": version,
            "version_sha256": file_sha256(version_path),
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "internal_test_rows": internal_test_rows,
        },
        "external_dataset": {
            "artifact_dir": str(external_corpus),
            "version": external_version,
            "version_sha256": file_sha256(external_version_path) if external_version_path.exists() else None,
            "rows": external_rows,
        },
        "selection_policy": selection_policy,
        "hyperparameters": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "learning_rate": learning_rate,
            "precision": precision,
            "ct2_quantization": quantization,
        },
        "resume": resume,
        "train_command": train_command,
        "export_command": export_command,
        "evaluations": evaluations,
    }


def _write_result(run_dir: Path, result: dict[str, Any]) -> None:
    """Persist terminal run state independently from the original immutable execution plan."""
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _lineage_signature(plan: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields that must never change when a failed experiment is resumed."""
    dataset = plan["dataset"]
    assert isinstance(dataset, dict)
    return {
        "language": plan["language"],
        "base_model": plan["base_model"],
        "profile_sha256": plan["profile_sha256"],
        "dataset_version_sha256": dataset["version_sha256"],
        "selection_policy": plan["selection_policy"],
        "hyperparameters": plan["hyperparameters"],
    }


def _assert_resume_lineage(run_dir: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Refuse to continue checkpoints if corpus/profile/hyperparameters differ from the run that created them."""
    plan_path = run_dir / "RUN_PLAN.json"
    if not plan_path.exists():
        raise RuntimeError(f"cannot resume experiment without original RUN_PLAN.json: {run_dir}")
    existing = _read_json_object(plan_path)
    if _lineage_signature(existing) != _lineage_signature(plan):
        raise RuntimeError(
            "resume lineage mismatch: existing checkpoints were created with a different corpus, profile, "
            "selection policy or hyperparameter set"
        )
    return existing


def _final_model_matches_plan(final_hf: Path, plan: dict[str, Any]) -> bool:
    """Trust an existing final model only when its recorded corpus fingerprint matches the current frozen plan."""
    experiment_path = final_hf / "experiment.json"
    if not experiment_path.exists():
        return False
    experiment = _read_json_object(experiment_path)
    dataset = plan["dataset"]
    assert isinstance(dataset, dict)
    version = dataset["version"]
    assert isinstance(version, dict)
    return (
        experiment.get("dataset_fingerprint_sha256") == version.get("fingerprint_sha256")
        and experiment.get("profile_sha256") == plan.get("profile_sha256")
        and experiment.get("base_model") == plan.get("base_model")
    )


def _execute_plan(plan: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    """Execute or resume a lineage-bound run while treating export/evaluation as reproducible derived phases."""
    run_dir = Path(str(plan["run_dir"]))
    final_hf = run_dir / "hf" / "final"
    ct2_dir = run_dir / "ct2"
    result_path = run_dir / "RUN_RESULT.json"

    if run_dir.exists() and any(run_dir.iterdir()):
        if result_path.exists():
            previous_result = _read_json_object(result_path)
            if previous_result.get("status") == "completed":
                _assert_resume_lineage(run_dir, plan)
                return previous_result
        if not resume:
            raise RuntimeError(f"refusing to overwrite an existing experiment: {run_dir}")
        existing_plan = _assert_resume_lineage(run_dir, plan)
        started_at = str(existing_plan.get("started_at") or _now())
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        started_at = _now()
        started = {**plan, "status": "running", "started_at": started_at}
        (run_dir / "RUN_PLAN.json").write_text(
            json.dumps(started, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    completed_evaluations: list[dict[str, Any]] = []
    try:
        # A failed export or evaluation must not force another multi-hour GPU training job. We only skip
        # training when the final Hugging Face model proves it belongs to this exact frozen corpus/profile.
        if not _final_model_matches_plan(final_hf, plan):
            subprocess.run([str(item) for item in plan["train_command"]], check=True)
            if not _final_model_matches_plan(final_hf, plan):
                raise RuntimeError("training exited without a lineage-matching final model")

        export_marker = run_dir / "CT2_EXPORT.json"
        if not export_marker.exists():
            # CTranslate2 is derived from the final HF checkpoint. A partial directory is safe to remove and
            # regenerate; the authoritative training checkpoint above is never deleted here.
            if ct2_dir.exists():
                shutil.rmtree(ct2_dir)
            subprocess.run([str(item) for item in plan["export_command"]], check=True)
            export_marker.write_text(
                json.dumps({"completed_at": _now(), "model": str(ct2_dir)}, indent=2),
                encoding="utf-8",
            )

        for evaluation in plan["evaluations"]:
            output = run_dir / f"{evaluation['name']}.json"
            if not output.exists():
                subprocess.run([str(item) for item in evaluation["command"]], check=True)
            completed_evaluations.append(
                {
                    "name": evaluation["name"],
                    "rows": evaluation["rows"],
                    "output": str(output),
                }
            )
    except BaseException as exc:
        failed = {
            **plan,
            "status": "failed",
            "started_at": started_at,
            "failed_at": _now(),
            "error": f"{type(exc).__name__}:{exc}",
            "completed_evaluations": completed_evaluations,
        }
        _write_result(run_dir, failed)
        raise

    result = {
        **plan,
        "status": "completed",
        "started_at": started_at,
        "completed_at": _now(),
        "completed_evaluations": completed_evaluations,
        "hf_final": str(final_hf),
        "ct2_model": str(ct2_dir),
    }
    _write_result(run_dir, result)
    return result


def main() -> None:
    """Plan or execute one or all frozen-corpus Whisper-small baselines."""
    parser = argparse.ArgumentParser(description="Plan or execute corpus-v0 Whisper-small ASR baselines")
    parser.add_argument("--language", choices=[*_LANGUAGES, "all"], default="all")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/bootstrap"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("training/configs/languages"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/asr"))
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--quantization",
        choices=["float16", "float32", "int8", "int8_float16", "int8_float32"],
        default="float16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--require-external-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    languages = list(_LANGUAGES) if args.language == "all" else [args.language]
    plans = [
        build_language_plan(
            language=language,
            artifacts_root=args.artifacts_root,
            profiles_dir=args.profiles_dir,
            output_root=args.output_root,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            precision=args.precision,
            quantization=args.quantization,
            require_external_eval=args.require_external_eval,
            device=args.device,
            compute_type=args.compute_type,
            resume=args.resume,
        )
        for language in languages
    ]
    if not args.execute:
        print(json.dumps({"status": "planned", "runs": plans}, ensure_ascii=False, indent=2))
        return

    results = [_execute_plan(plan, resume=args.resume) for plan in plans]
    print(json.dumps({"status": "completed", "runs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
