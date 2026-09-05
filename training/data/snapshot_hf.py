"""Reproducible Hugging Face snapshot downloader with catalog and revision-pin enforcement."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from training.data.catalog import SourceCatalog


def main() -> None:
    """Download one approved dataset snapshot and persist an immutable receipt next to raw files."""
    parser = argparse.ArgumentParser(description="Download a policy-approved Hugging Face dataset snapshot")
    parser.add_argument("--catalog", type=Path, default=Path("training/configs/source_catalog.yaml"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--usage", choices=["production", "evaluation", "research"], required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = SourceCatalog(args.catalog)
    source = catalog.get(args.source)
    if args.language not in source.languages:
        raise SystemExit(f"source {source.source_id} does not declare language {args.language}")
    if not source.allows(args.usage):
        raise SystemExit(f"source {source.source_id} is not approved for {args.usage} usage")
    if source.provider != "huggingface" or not source.repo_id:
        raise SystemExit(f"source {source.source_id} is not a Hugging Face snapshot source")
    if source.requires_revision_pin and not args.revision:
        raise SystemExit("reviewable snapshots must use an explicit --revision commit SHA")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before downloading external snapshots") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
    )
    receipt = {
        "schema_version": 1,
        "source": asdict(source),
        "language": args.language,
        "requested_usage": args.usage,
        "requested_revision": args.revision,
        "resolved_path": str(resolved),
        "downloaded_at": datetime.now(UTC).isoformat(),
    }
    (args.output / "SNAPSHOT_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
