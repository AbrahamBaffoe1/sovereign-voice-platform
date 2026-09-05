"""Bulk importer for consented first-party voice messages and team recordings."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from app.services.corpus_audio import normalize_clip, segment_recording
from app.services.corpus_store import CorpusStore

_REQUIRED = {"path", "source_id", "speaker", "language", "multi_speaker", "consent_attested"}
_TRUE = {"1", "true", "yes", "y"}


def _truth(value: str | None) -> bool:
    """Parse conservative inventory booleans; unfamiliar values never imply consent."""
    return (value or "").strip().casefold() in _TRUE


def import_inventory(
    *,
    inventory: Path,
    input_root: Path,
    corpus_root: Path,
    max_recording_seconds: float = 1800.0,
    target_rate: int = 16000,
) -> dict[str, object]:
    """Import single-speaker rows and quarantine multi-speaker recordings for diarization."""
    store = CorpusStore(corpus_root)
    imported: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    with inventory.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(_REQUIRED - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"inventory is missing required columns: {missing}")
        for line_no, row in enumerate(reader, 2):
            relative = (row.get("path") or "").strip()
            source_id = (row.get("source_id") or "").strip()
            speaker = (row.get("speaker") or "").strip()
            language = (row.get("language") or "").strip()
            dialect = (row.get("dialect") or "").strip() or None
            multi_speaker = _truth(row.get("multi_speaker"))
            consent = _truth(row.get("consent_attested"))
            path = (input_root / relative).resolve()
            if not consent:
                rejected.append({"line": line_no, "path": relative, "reason": "consent_not_attested"})
                continue
            if not path.exists() or not path.is_file():
                rejected.append({"line": line_no, "path": relative, "reason": "missing_file"})
                continue
            if multi_speaker:
                quarantined.append(
                    {
                        "line": line_no,
                        "path": str(path),
                        "source_id": source_id,
                        "language": language,
                        "reason": "requires_diarization",
                    }
                )
                continue
            payload = path.read_bytes()
            try:
                segments = segment_recording(
                    payload, max_seconds=max_recording_seconds, target_rate=target_rate
                )
                if not segments:
                    segments = [
                        normalize_clip(
                            payload,
                            max_seconds=max_recording_seconds,
                            target_rate=target_rate,
                        )
                    ]
                for index, segment in enumerate(segments):
                    item = store.create_item(
                        wav_bytes=segment.wav_bytes,
                        language=language,
                        speaker=speaker,
                        source_id=f"{source_id}#segment-{index:04d}",
                        consent_attested=True,
                        duration_seconds=segment.duration_seconds,
                        sample_rate=segment.sample_rate,
                        dialect=dialect,
                        parent_source_id=source_id,
                        segment_index=index,
                    )
                    imported.append(asdict(item))
            except Exception as exc:
                rejected.append(
                    {"line": line_no, "path": relative, "reason": f"{type(exc).__name__}:{exc}"}
                )
    return {
        "imported_segments": imported,
        "quarantined": quarantined,
        "rejected": rejected,
        "counts": {
            "imported_segments": len(imported),
            "quarantined": len(quarantined),
            "rejected": len(rejected),
        },
    }


def main() -> None:
    """Run the inventory importer and print a machine-readable batch report."""
    parser = argparse.ArgumentParser(description="Import consented first-party team recordings")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path("data/corpus"))
    parser.add_argument("--max-recording-seconds", type=float, default=1800.0)
    parser.add_argument("--target-rate", type=int, default=16000)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = import_inventory(
        inventory=args.inventory,
        input_root=args.input_root,
        corpus_root=args.corpus_root,
        max_recording_seconds=args.max_recording_seconds,
        target_rate=args.target_rate,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
