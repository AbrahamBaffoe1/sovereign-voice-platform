"""Dataset compiler that validates recordings, rejects unsafe rows, freezes deterministic splits, and emits corpus inventories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.common.audio_quality import inspect_audio
from training.common.manifest import SpeechRecord, file_sha256, normalize_transcript, stable_partition, write_jsonl


def parse_args() -> argparse.Namespace:
    """Expose corpus input paths and conservative acceptance thresholds used by the deterministic dataset compiler."""
    parser = argparse.ArgumentParser(description="Build validated NeMo/HF speech manifests from CSV")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with audio,text[,speaker] columns")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--allow-suspicious", action="store_true")
    parser.add_argument("--allow-multichannel", action="store_true")
    parser.add_argument("--required-sample-rate", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Compile raw CSV rows into trustworthy speech manifests: verify files/text, measure audio, reject duplicates and quality violations, create stable splits, and emit rejection plus grapheme-inventory artifacts for review."""
    args = parse_args()
    splits: dict[str, list[SpeechRecord]] = defaultdict(list)
    rejected: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    char_counter: Counter[str] = Counter()
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio", "text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"CSV must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, 2):
            rel = (row.get("audio") or "").strip()
            text = normalize_transcript(row.get("text") or "")
            speaker = (row.get("speaker") or "").strip() or None
            audio_path = (args.audio_root / rel).resolve()
            reasons: list[str] = []
            if not rel or not audio_path.exists(): reasons.append("missing_audio")
            if not text: reasons.append("empty_text")
            if reasons:
                rejected.append({"row": row_number, "audio": rel, "reasons": reasons}); continue
            try:
                quality = inspect_audio(audio_path); digest = file_sha256(audio_path)
            except Exception as exc:
                rejected.append({"row": row_number, "audio": rel, "reasons": [f"audio_error:{exc}"]}); continue
            if digest in seen_hashes: reasons.append("duplicate_audio")
            if quality.channels != 1 and not args.allow_multichannel: reasons.append("multichannel_audio")
            if args.required_sample_rate and quality.sample_rate != args.required_sample_rate: reasons.append(f"sample_rate:{quality.sample_rate}")
            if quality.duration < args.min_seconds: reasons.append("too_short")
            if quality.duration > args.max_seconds: reasons.append("too_long")
            if quality.suspicious and not args.allow_suspicious: reasons.append("quality_flag")
            if reasons:
                rejected.append({"row": row_number, "audio": rel, "reasons": reasons}); continue
            seen_hashes.add(digest); char_counter.update(text)
            record = SpeechRecord(audio_filepath=str(audio_path), text=text, duration=quality.duration, speaker=speaker, language=args.language, sha256=digest)
            splits[stable_partition(digest)].append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(args.output / f"{split}.json", splits.get(split, []), nemo=True)
    (args.output / "rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = {"language": args.language,"accepted": sum(len(rows) for rows in splits.values()),"rejected": len(rejected),"splits": {name: len(rows) for name, rows in splits.items()},"characters": [{"char": char, "count": count} for char, count in char_counter.most_common()]}
    (args.output / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in inventory.items() if k != "characters"}, indent=2))


if __name__ == "__main__":
    main()
