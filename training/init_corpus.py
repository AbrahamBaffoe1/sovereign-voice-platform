"""Create a policy-bound corpus workspace for one language without inventing sample training data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.common.language_profile import load_language_profile

BASE_COLUMNS = ["audio", "text", "speaker", "dialect", "source_id", "consent_attested", "transcript_reviewed"]


def parse_args() -> argparse.Namespace:
    """Accept a language profile and destination directory for a new corpus workspace."""
    parser = argparse.ArgumentParser(description="Initialize a strict speech-corpus workspace")
    parser.add_argument("--profile", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def initialize_corpus(profile_path: Path, output: Path) -> None:
    """Create metadata headers, an audio directory and a machine-readable copy of collection policy."""
    profile = load_language_profile(profile_path); output.mkdir(parents=True, exist_ok=True)
    wavs = output / "wavs"; wavs.mkdir(exist_ok=True); (wavs / ".gitkeep").touch(exist_ok=True)
    metadata = output / "metadata.csv"
    if not metadata.exists():
        with metadata.open("w", encoding="utf-8", newline="") as handle: csv.writer(handle).writerow(BASE_COLUMNS)
    policy = {"language": profile.code, "name": profile.name, "iso639_3": profile.iso639_3, "script": profile.script, "required_metadata": list(profile.corpus.required_metadata), "require_consent": profile.corpus.require_consent, "require_reviewed_transcript": profile.corpus.require_reviewed_transcript, "split_unit": profile.corpus.split_unit, "duration_seconds": {"min": profile.corpus.min_seconds, "max": profile.corpus.max_seconds}, "profile": str(profile_path), "note": "Do not mark transcript_reviewed=true until a human reviewer has checked the recording against the transcript. Do not mark consent_attested=true without collection evidence."}
    (output / "CORPUS_POLICY.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Initialize the requested corpus workspace and print its path for shell automation."""
    args = parse_args(); initialize_corpus(args.profile, args.output); print(args.output)


if __name__ == "__main__":
    main()
