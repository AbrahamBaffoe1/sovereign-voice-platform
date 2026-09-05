"""Versioned ASR experiment plans separating scientific inputs from cluster execution details."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Immutable ASR identity derived from language, dataset fingerprint, candidate, and seed."""

    experiment_id: str
    language: str
    dataset_id: str
    dataset_fingerprint: str
    candidate: str
    family: str
    base_model: str | None
    seed: int
    train_manifest: str
    validation_manifest: str
    output_dir: str
    trainable: bool
    blocked_reason: str | None


def build_plan(
    *,
    language: str,
    dataset_version: Path,
    candidate: str,
    candidates_config: Path,
    artifacts_dir: Path,
    seed: int = 17,
) -> ExperimentPlan:
    """Create a deterministic experiment plan and fail before GPU allocation on malformed metadata."""
    dataset = json.loads(dataset_version.read_text(encoding="utf-8"))
    config = yaml.safe_load(candidates_config.read_text(encoding="utf-8")) or {}
    candidates = config.get("candidates") or {}
    if candidate not in candidates:
        raise KeyError(f"unknown ASR candidate: {candidate}")
    raw = candidates[candidate]
    if not isinstance(raw, dict):
        raise ValueError(f"candidate {candidate!r} must be a mapping")
    dataset_id = str(dataset.get("dataset_id") or "").strip()
    fingerprint = str(dataset.get("fingerprint_sha256") or "").strip()
    if not dataset_id or not fingerprint:
        raise ValueError("dataset_version.json is missing dataset_id/fingerprint_sha256")
    identity = json.dumps(
        {
            "language": language,
            "dataset_id": dataset_id,
            "fingerprint": fingerprint,
            "candidate": candidate,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    experiment_id = f"{language}-{candidate}-{suffix}"
    prepared = dataset_version.parent
    return ExperimentPlan(
        experiment_id=experiment_id,
        language=language,
        dataset_id=dataset_id,
        dataset_fingerprint=fingerprint,
        candidate=candidate,
        family=str(raw.get("family", "unknown")),
        base_model=str(raw["base_model"]) if raw.get("base_model") else None,
        seed=seed,
        train_manifest=str(prepared / "train.json"),
        validation_manifest=str(prepared / "validation.json"),
        output_dir=str(artifacts_dir / experiment_id),
        trainable=bool(raw.get("trainable", False)),
        blocked_reason=str(raw["blocked_reason"]) if raw.get("blocked_reason") else None,
    )


def write_plan(plan: ExperimentPlan, output: Path) -> None:
    """Persist the exact plan consumed by local, Slurm, and Kubernetes runners."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")


def read_plan(path: Path) -> ExperimentPlan:
    """Load a frozen experiment plan without recomputing its identity."""
    return ExperimentPlan(**json.loads(path.read_text(encoding="utf-8")))
