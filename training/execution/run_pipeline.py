"""Orchestrate the real corpus -> TTS review -> ASR training pipeline on persistent storage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.execution.environment import WorkspaceLayout, execution_lock, preflight

_LANGUAGES = ("tw", "gaa", "ee", "ha")


def _now() -> str:
    """Return an offset-aware timestamp for durable execution state."""
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace state atomically so a killed process cannot leave half-written JSON as the next run's truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _stream_command(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    """Stream child output to both the operator console and a persistent log without buffering the full job."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now()}] $ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def _phase(
    *,
    name: str,
    command: list[str],
    state: dict[str, Any],
    state_path: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    """Persist phase transitions before and after execution so an interrupted machine shows exactly where it stopped."""
    phases = state.setdefault("phases", {})
    assert isinstance(phases, dict)
    phase_state = {"status": "running", "started_at": _now(), "command": command}
    phases[name] = phase_state
    _atomic_json(state_path, state)
    try:
        _stream_command(command, env=env, log_path=log_path)
    except BaseException as exc:
        phase_state.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error": f"{type(exc).__name__}:{exc}",
            }
        )
        state["status"] = "failed"
        state["failed_phase"] = name
        _atomic_json(state_path, state)
        raise
    phase_state.update({"status": "completed", "completed_at": _now()})
    _atomic_json(state_path, state)


def _selected_languages(value: str) -> list[str]:
    """Expand the CLI's all token once so every downstream command sees an explicit language."""
    return list(_LANGUAGES) if value == "all" else [value]


def _corpus_command(args: argparse.Namespace, layout: WorkspaceLayout) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "training.data.bootstrap",
        "--language",
        args.language,
        "--task",
        "both",
        "--include-eval",
        "--data-root",
        str(layout.data_root),
        "--artifacts-root",
        str(layout.artifacts_root),
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    if args.force_reacquire:
        command.append("--force-reacquire")
    if args.refresh_source_locks:
        command.append("--refresh-lock")
    return command


def _tts_command(language: str, layout: WorkspaceLayout) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training.tts.readiness",
        "--language",
        language,
        "--artifacts-root",
        str(layout.artifacts_root),
        "--output-root",
        str(layout.tts_readiness_root),
    ]


def _asr_command(args: argparse.Namespace, language: str, layout: WorkspaceLayout) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "training.asr.run_baseline",
        "--language",
        language,
        "--artifacts-root",
        str(layout.artifacts_root),
        "--output-root",
        str(layout.experiments_root),
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--learning-rate",
        str(args.learning_rate),
        "--precision",
        args.precision,
        "--quantization",
        args.quantization,
        "--resume",
        "--execute",
    ]
    if args.require_external_eval:
        command.append("--require-external-eval")
    return command


def parse_args() -> argparse.Namespace:
    """Expose only decisions an operator should make; corpus/model policy remains in version-controlled profiles."""
    parser = argparse.ArgumentParser(description="Run the persistent sovereign voice training pipeline")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("VOICE_EXECUTION_ROOT", "/srv/sovereign-voice")),
    )
    parser.add_argument("--language", choices=[*_LANGUAGES, "all"], default="all")
    parser.add_argument("--min-free-gb", type=float, default=150.0)
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
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--skip-tts-readiness", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--require-external-eval", action="store_true")
    parser.add_argument("--force-reacquire", action="store_true")
    parser.add_argument("--refresh-source-locks", action="store_true")
    parser.add_argument(
        "--allow-ephemeral",
        action="store_true",
        help="Smoke-test escape hatch. Do not use it for a real corpus/model build.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute phases under one workspace lock and preserve enough state to resume after machine interruption."""
    args = parse_args()
    layout = WorkspaceLayout.from_root(args.workspace)
    environment = preflight(
        layout,
        min_free_gb=args.min_free_gb,
        require_gpu=not args.skip_asr,
        allow_ephemeral=args.allow_ephemeral,
    )
    child_env = layout.child_environment()
    state_path = layout.state_root / "REAL_EXECUTION.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _now(),
        "language": args.language,
        "workspace": str(layout.root),
        "environment_report": str(layout.state_root / "EXECUTION_ENVIRONMENT.json"),
        "environment": environment,
        "phases": {},
    }
    _atomic_json(state_path, state)

    with execution_lock(layout):
        log_path = layout.logs_root / "real-execution.log"
        if not args.skip_corpus:
            _phase(
                name="corpus_v0",
                command=_corpus_command(args, layout),
                state=state,
                state_path=state_path,
                env=child_env,
                log_path=log_path,
            )

        # TTS remains review-gated. This phase intentionally generates evidence and blockers; it never
        # turns observed Unicode characters into an approved grapheme/G2P system by itself.
        if not args.skip_tts_readiness:
            for language in _selected_languages(args.language):
                _phase(
                    name=f"tts_readiness_{language}",
                    command=_tts_command(language, layout),
                    state=state,
                    state_path=state_path,
                    env=child_env,
                    log_path=log_path,
                )

        # ASR can proceed independently because its tokenizer policy is already explicit in the language
        # profiles. Each language is launched separately so one upstream benchmark problem does not erase
        # successful checkpoints from the other languages.
        if not args.skip_asr:
            for language in _selected_languages(args.language):
                _phase(
                    name=f"asr_baseline_{language}",
                    command=_asr_command(args, language, layout),
                    state=state,
                    state_path=state_path,
                    env=child_env,
                    log_path=log_path,
                )

    state.update({"status": "completed", "completed_at": _now()})
    _atomic_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
