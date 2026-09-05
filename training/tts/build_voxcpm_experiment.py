"""Build a frozen VoxCPM2 adaptation plan without launching vendor training code implicitly."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Mode = Literal["lora", "full"]


@dataclass(frozen=True, slots=True)
class VoxCPMExperiment:
    """Immutable description of a target-language VoxCPM2 adaptation experiment."""

    experiment_id: str
    language: str
    dataset_id: str
    dataset_fingerprint: str
    base_model: str
    mode: Mode
    train_manifest: str
    validation_manifest: str
    output_dir: str
    seed: int


def build_experiment(
    *,
    language: str,
    dataset_version: Path,
    mode: Mode,
    output_root: Path,
    base_model: str = "openbmb/VoxCPM2",
    seed: int = 17,
) -> VoxCPMExperiment:
    """Tie one adaptation plan to an immutable dataset fingerprint and explicit training mode."""
    payload = json.loads(dataset_version.read_text(encoding="utf-8"))
    dataset_id = str(payload.get("dataset_id") or "").strip()
    fingerprint = str(payload.get("fingerprint_sha256") or "").strip()
    if not dataset_id or not fingerprint:
        raise ValueError("dataset_version.json is missing dataset_id/fingerprint_sha256")
    identity = json.dumps(
        {
            "language": language,
            "dataset_id": dataset_id,
            "fingerprint": fingerprint,
            "base_model": base_model,
            "mode": mode,
            "seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    experiment_id = f"{language}-voxcpm2-{mode}-{suffix}"
    prepared = dataset_version.parent
    return VoxCPMExperiment(
        experiment_id=experiment_id,
        language=language,
        dataset_id=dataset_id,
        dataset_fingerprint=fingerprint,
        base_model=base_model,
        mode=mode,
        train_manifest=str(prepared / "train.json"),
        validation_manifest=str(prepared / "validation.json"),
        output_dir=str(output_root / experiment_id),
        seed=seed,
    )


def main() -> None:
    """Write a reviewable experiment JSON mapped later to a pinned VoxCPM trainer revision."""
    parser = argparse.ArgumentParser(description="Freeze a VoxCPM2 target-language adaptation plan")
    parser.add_argument("--language", required=True)
    parser.add_argument("--dataset-version", type=Path, required=True)
    parser.add_argument("--mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--base-model", default="openbmb/VoxCPM2")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/tts"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_experiment(
        language=args.language,
        dataset_version=args.dataset_version,
        mode=args.mode,
        output_root=args.output_root,
        base_model=args.base_model,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
