"""Launch reproducible corpus-v0 Whisper baselines without contaminating external benchmarks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.common.language_profile import load_language_profile
from training.common.manifest import file_sha256

_LANGUAGES = ("tw", "gaa", "ee", "ha")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _safe_model_token(model: str) -> str:
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
    internal_test_rows = sum(
        1
        for line in audit.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("split") == "test"
    ) if audit.exists() else 0
    if train_rows < 1:
        raise ValueError(f"frozen corpus has no training rows: {train}")

    external_corpus = artifacts_root / language / "asr" / "corpus-v0-eval"
    external_version_path = external_corpus / "dataset_version.json"
    external_audit = external_corpus / "audit.jsonl"
    external_rows = _jsonl_rows(external_audit)
    external_version = _read_json_object(external_version_path) if external_version_path.exists() else None
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
        "schema_version": 1,
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
        "train_command": train_command,
        "export_command": export_command,
        "evaluations": evaluations,
    }


def _execute_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a prevalidated run sequentially and leave durable lineage before the first GPU operation."""
    run_dir = Path(str(plan["run_dir"]))
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite an existing experiment: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = {**plan, "status": "running", "started_at": _now()}
    (run_dir / "RUN_PLAN.json").write_text(json.dumps(started, ensure_ascii=False, indent=2), encoding="utf-8")

    subprocess.run([str(item) for item in plan["train_command"]], check=True)
    subprocess.run([str(item) for item in plan["export_command"]], check=True)
    completed_evaluations: list[dict[str, Any]] = []
    for evaluation in plan["evaluations"]:
        subprocess.run([str(item) for item in evaluation["command"]], check=True)
        completed_evaluations.append(
            {
                "name": evaluation["name"],
                "rows": evaluation["rows"],
                "output": str(run_dir / f"{evaluation['name']}.json"),
            }
        )

    result = {
        **plan,
        "status": "completed",
        "started_at": started["started_at"],
        "completed_at": _now(),
        "completed_evaluations": completed_evaluations,
        "hf_final": str(run_dir / "hf" / "final"),
        "ct2_model": str(run_dir / "ct2"),
    }
    (run_dir / "RUN_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
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
        )
        for language in languages
    ]
    if not args.execute:
        print(json.dumps({"status": "planned", "runs": plans}, ensure_ascii=False, indent=2))
        return

    results = [_execute_plan(plan) for plan in plans]
    print(json.dumps({"status": "completed", "runs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
