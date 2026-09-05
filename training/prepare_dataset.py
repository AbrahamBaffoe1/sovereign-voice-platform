"""Strict multilingual dataset compiler for ASR/TTS corpora and auditable split artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.common.audio_quality import inspect_audio
from training.common.corpus_quality import build_quality_report
from training.common.language_profile import LanguageTrainingProfile, load_language_profile
from training.common.manifest import (
    SpeechRecord,
    dataset_fingerprint,
    file_sha256,
    normalize_transcript,
    stable_partition,
    write_jsonl,
)

_TRUE = {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    """Expose raw corpus paths plus an optional strict language profile that owns acceptance policy."""
    parser = argparse.ArgumentParser(description="Build validated NeMo/HF speech manifests from CSV")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--language", default=None, help="Legacy mode only when --profile is omitted")
    parser.add_argument("--min-seconds", type=float, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--allow-suspicious", action="store_true")
    parser.add_argument("--allow-multichannel", action="store_true")
    parser.add_argument("--required-sample-rate", type=int, default=None)
    return parser.parse_args()


def _bool(value: str | None) -> bool:
    """Parse conservative CSV booleans; blank or unfamiliar values are false rather than trusted."""
    return (value or "").strip().casefold() in _TRUE


def _resolve_policy(
    args: argparse.Namespace,
) -> tuple[str, LanguageTrainingProfile | None, float, float]:
    """Resolve language and duration policy from the profile while preserving a legacy CLI path."""
    profile = load_language_profile(args.profile) if args.profile else None
    if profile:
        if args.language and args.language != profile.code:
            raise SystemExit(
                f"--language {args.language!r} does not match profile language {profile.code!r}"
            )
        return (
            profile.code,
            profile,
            args.min_seconds if args.min_seconds is not None else profile.corpus.min_seconds,
            args.max_seconds if args.max_seconds is not None else profile.corpus.max_seconds,
        )
    if not args.language:
        raise SystemExit("either --profile or --language is required")
    return (
        args.language,
        None,
        args.min_seconds if args.min_seconds is not None else 0.5,
        args.max_seconds if args.max_seconds is not None else 20.0,
    )


def main() -> None:
    """Validate recordings and governance metadata, create stable splits, and emit audit artifacts."""
    args = parse_args()
    language, profile, min_seconds, max_seconds = _resolve_policy(args)
    splits: dict[str, list[SpeechRecord]] = defaultdict(list)
    accepted: list[SpeechRecord] = []
    rejected: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    char_counter: Counter[str] = Counter()

    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio", "text"}
        if profile:
            required.update(profile.corpus.required_metadata)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise SystemExit(f"CSV is missing required columns: {missing}")

        for row_number, row in enumerate(reader, 2):
            rel = (row.get("audio") or "").strip()
            text = normalize_transcript(row.get("text") or "")
            speaker = (row.get("speaker") or "").strip() or None
            dialect = (row.get("dialect") or "").strip() or None
            source_id = (row.get("source_id") or "").strip() or None
            consent_attested = _bool(row.get("consent_attested"))
            transcript_reviewed = _bool(row.get("transcript_reviewed"))
            audio_path = (args.audio_root / rel).resolve()
            reasons: list[str] = []

            if not rel or not audio_path.exists():
                reasons.append("missing_audio")
            if not text:
                reasons.append("empty_text")
            if profile:
                if profile.corpus.split_unit == "speaker" and not speaker:
                    reasons.append("missing_speaker")
                if profile.corpus.require_consent and not consent_attested:
                    reasons.append("consent_not_attested")
                if profile.corpus.require_reviewed_transcript and not transcript_reviewed:
                    reasons.append("transcript_not_reviewed")
                if "dialect" in profile.corpus.required_metadata and not dialect:
                    reasons.append("missing_dialect")
                if "source_id" in profile.corpus.required_metadata and not source_id:
                    reasons.append("missing_source_id")
            if reasons:
                rejected.append({"row": row_number, "audio": rel, "reasons": reasons})
                continue

            try:
                quality = inspect_audio(audio_path)
                digest = file_sha256(audio_path)
            except Exception as exc:
                rejected.append(
                    {"row": row_number, "audio": rel, "reasons": [f"audio_error:{exc}"]}
                )
                continue

            if digest in seen_hashes:
                reasons.append("duplicate_audio")
            require_mono = profile.corpus.require_mono if profile else not args.allow_multichannel
            if quality.channels != 1 and require_mono and not args.allow_multichannel:
                reasons.append("multichannel_audio")
            if args.required_sample_rate and quality.sample_rate != args.required_sample_rate:
                reasons.append(f"sample_rate:{quality.sample_rate}")
            if quality.duration < min_seconds:
                reasons.append("too_short")
            if quality.duration > max_seconds:
                reasons.append("too_long")
            if quality.suspicious and not args.allow_suspicious:
                reasons.append("quality_flag")
            if reasons:
                rejected.append({"row": row_number, "audio": rel, "reasons": reasons})
                continue

            seen_hashes.add(digest)
            char_counter.update(text)
            split_key = speaker if profile and profile.corpus.split_unit == "speaker" else digest
            assert split_key is not None
            split = stable_partition(split_key)
            record = SpeechRecord(
                audio_filepath=str(audio_path),
                text=text,
                duration=quality.duration,
                speaker=speaker,
                language=language,
                sha256=digest,
                dialect=dialect,
                source_id=source_id,
                consent_attested=consent_attested if profile else None,
                transcript_reviewed=transcript_reviewed if profile else None,
                split=split,
            )
            splits[split].append(record)
            accepted.append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        write_jsonl(args.output / f"{split}.json", splits.get(split, []), nemo=True)

    # The audit manifest is intentionally richer than model manifests. It lets us prove which source,
    # consent status, dialect and review state produced every training row.
    write_jsonl(args.output / "audit.jsonl", accepted, nemo=False)

    (args.output / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inventory = {
        "language": language,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "split_unit": profile.corpus.split_unit if profile else "audio",
        "characters": [
            {"char": char, "count": count} for char, count in char_counter.most_common()
        ],
    }
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    quality_report = build_quality_report(accepted).as_dict()
    quality_report.update({"language": language, "rejected": len(rejected)})
    (args.output / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fingerprint = dataset_fingerprint(accepted)
    profile_hash = file_sha256(args.profile) if args.profile else None
    dataset_version = {
        "language": language,
        "dataset_id": f"{language}-{fingerprint[:16]}",
        "fingerprint_sha256": fingerprint,
        "profile": str(args.profile) if args.profile else None,
        "profile_sha256": profile_hash,
        "accepted": len(accepted),
        "hours": quality_report["hours"],
    }
    (args.output / "dataset_version.json").write_text(
        json.dumps(dataset_version, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if profile and profile.corpus.split_unit == "speaker" and quality_report["speaker_leakage"]:
        raise SystemExit("speaker leakage detected despite speaker-disjoint split policy")
    print(json.dumps(quality_report, indent=2))


if __name__ == "__main__":
    main()
