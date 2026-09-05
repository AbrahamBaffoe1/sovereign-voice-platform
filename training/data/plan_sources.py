"""CLI that explains which governed sources may be used before any network download occurs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.data.catalog import SourceCatalog


def main() -> None:
    """Print the policy-approved source plan for one language and intended usage."""
    parser = argparse.ArgumentParser(description="Plan governed speech-data sources")
    parser.add_argument("--catalog", type=Path, default=Path("training/configs/source_catalog.yaml"))
    parser.add_argument("--language", required=True)
    parser.add_argument("--usage", choices=["production", "evaluation", "research"], required=True)
    args = parser.parse_args()
    sources = SourceCatalog(args.catalog).plan(language=args.language, usage=args.usage)
    print(json.dumps([asdict(source) for source in sources], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
