"""Execute or render a frozen ASR experiment for local, Slurm, or Kubernetes environments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from training.experiments.plans import ExperimentPlan, read_plan


def command_for(plan: ExperimentPlan) -> list[str]:
    """Translate a supported plan into the canonical trainer command without changing inputs."""
    if not plan.trainable:
        raise RuntimeError(plan.blocked_reason or f"candidate {plan.candidate} has no implemented trainer")
    if plan.family != "whisper" or not plan.base_model:
        raise RuntimeError(f"trainer for family {plan.family!r} is not implemented")
    return [
        "python",
        "-m",
        "training.asr.finetune_whisper",
        "--train",
        plan.train_manifest,
        "--validation",
        plan.validation_manifest,
        "--base-model",
        plan.base_model,
        "--output",
        plan.output_dir,
    ]


def render_slurm(plan: ExperimentPlan, *, gpus: int = 1, cpus: int = 8, memory_gb: int = 48) -> str:
    """Render a Slurm script invoking the same frozen command as local execution."""
    command = shlex.join(command_for(plan))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={plan.experiment_id}",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --cpus-per-task={cpus}",
            f"#SBATCH --mem={memory_gb}G",
            "#SBATCH --output=logs/%x-%j.log",
            "set -euo pipefail",
            command,
            "",
        ]
    )


def render_kubernetes(plan: ExperimentPlan, *, image: str, gpu_limit: int = 1) -> str:
    """Render a batch/v1 Kubernetes Job with the command embedded as an argument array."""
    payload = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": plan.experiment_id[:63]},
        "spec": {
            "backoffLimit": 0,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "trainer",
                            "image": image,
                            "command": command_for(plan),
                            "resources": {"limits": {"nvidia.com/gpu": gpu_limit}},
                        }
                    ],
                }
            },
        },
    }
    return json.dumps(payload, indent=2)


def main() -> None:
    """Execute a plan locally or emit cluster input without mutating the plan itself."""
    parser = argparse.ArgumentParser(description="Run or render a frozen ASR experiment")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--backend", choices=["local", "slurm", "kubernetes"], required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image", default="sovereign-voice-platform:training")
    args = parser.parse_args()
    plan = read_plan(args.plan)
    if args.backend == "local":
        subprocess.run(command_for(plan), check=True)
        return
    rendered = render_slurm(plan) if args.backend == "slurm" else render_kubernetes(plan, image=args.image)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
