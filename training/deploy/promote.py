"""CLI for explicit model promotion and rollback operations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.deploy.model_registry import ModelRegistry


def main() -> None:
    """Apply a promotion or rollback against the local immutable model registry."""
    parser = argparse.ArgumentParser(description="Promote or rollback a registered speech model")
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--task", choices=["asr", "tts"], required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--stage", choices=["staging", "production", "retired"], default="production")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    registry = ModelRegistry(args.root)
    if args.rollback:
        pointer = registry.rollback(task=args.task, language=args.language, stage=args.stage)
    else:
        if not args.model_id:
            raise SystemExit("--model-id is required unless --rollback is used")
        pointer = registry.promote(
            task=args.task, language=args.language, model_id=args.model_id, stage=args.stage
        )
    print(json.dumps(asdict(pointer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
