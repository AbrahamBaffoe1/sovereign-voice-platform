"""Evidence-driven TTS readiness gate for frozen custom-language corpus-v0 artifacts.

Observed corpus characters are review evidence, never an automatically approved alphabet. This
module intentionally does not infer pronunciations, create a G2P system, or silently promote an
experimental frontend into production policy.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from training.common.language_profile import LanguageTrainingProfile, load_language_profile
from training.common.manifest import file_sha256

_LANGUAGES = ("tw", "gaa", "ee", "ha")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a required JSON object with an explicit path-level error."""
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _iter_jsonl(path: Path):
    """Yield large audit manifests one row at a time so preflight remains bounded-memory."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield payload


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _observed_characters(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Render observed corpus characters as human-review evidence, not a tokenizer decision."""
    raw = inventory.get("characters") or []
    if not isinstance(raw, list):
        raise ValueError("inventory.characters must be a list")
    result: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        char = str(entry.get("char") or "")
        if not char:
            continue
        result.append(
            {
                "char": char,
                "count": int(entry.get("count", 0)),
                "codepoints": [f"U+{ord(item):04X}" for item in char],
                "unicode_names": [unicodedata.name(item, "UNNAMED") for item in char],
                "unicode_categories": [unicodedata.category(item) for item in char],
                "is_whitespace": char.isspace(),
            }
        )
    return result


def _audit_summary(path: Path) -> tuple[int, list[dict[str, Any]]]:
    """Summarize source contribution while preserving source license and revision boundaries."""
    counts: Counter[tuple[str, str, str]] = Counter()
    seconds: Counter[tuple[str, str, str]] = Counter()
    speakers: dict[tuple[str, str, str], set[str]] = {}
    total = 0
    for row in _iter_jsonl(path):
        total += 1
        source_id = str(row.get("source_id") or "unknown")
        source_family = source_id.split(":", 1)[0]
        license_name = str(row.get("source_license") or "unknown")
        revision = str(row.get("source_revision") or "unknown")
        key = (source_family, license_name, revision)
        counts[key] += 1
        seconds[key] += float(row.get("duration") or 0.0)
        speaker = str(row.get("speaker") or "").strip()
        if speaker:
            speakers.setdefault(key, set()).add(speaker)
    summary = [
        {
            "source": source,
            "license": license_name,
            "revision": revision,
            "rows": counts[(source, license_name, revision)],
            "hours": round(seconds[(source, license_name, revision)] / 3600.0, 6),
            "speakers": len(speakers.get((source, license_name, revision), set())),
        }
        for source, license_name, revision in sorted(counts)
    ]
    return total, summary


def _blockers(
    *,
    profile: LanguageTrainingProfile,
    accepted_rows: int,
    train_rows: int,
    validation_rows: int,
    quality: dict[str, Any],
) -> list[dict[str, str]]:
    """Return every unresolved production decision instead of hiding it behind permissive defaults."""
    blockers: list[dict[str, str]] = []
    if accepted_rows < 1 or train_rows < 1:
        blockers.append({"code": "empty_training_corpus", "detail": "Frozen TTS corpus has no trainable samples."})
    if quality.get("speaker_leakage"):
        blockers.append(
            {
                "code": "speaker_leakage",
                "detail": f"Speaker identities cross generated splits: {quality['speaker_leakage']}",
            }
        )
    if profile.tts.frontend == "experiment":
        blockers.append(
            {
                "code": "frontend_unselected",
                "detail": "Choose and review grapheme or phoneme frontend from corpus and native-speaker evidence.",
            }
        )
    if not profile.tts.tokenizer_reviewed:
        blockers.append(
            {
                "code": "tokenizer_unreviewed",
                "detail": "TTS tokenizer policy has not been explicitly reviewed and approved.",
            }
        )
    if not profile.reviewed_graphemes:
        blockers.append(
            {
                "code": "graphemes_unreviewed",
                "detail": "No reviewed grapheme inventory is frozen in the language profile.",
            }
        )
    if profile.tts.frontend == "phoneme" and not profile.tts.g2p_reviewed:
        blockers.append(
            {
                "code": "g2p_unreviewed",
                "detail": "Phoneme frontend requires reviewed lexicon/G2P and explicit OOV policy.",
            }
        )
    if validation_rows < 1:
        blockers.append(
            {
                "code": "no_internal_validation",
                "detail": "No internal TTS validation rows exist; define an uncontaminated validation strategy before training.",
            }
        )
    return blockers


def build_readiness_report(
    *,
    profile_path: Path,
    artifacts: Path,
) -> dict[str, Any]:
    """Build a durable review packet tied to one immutable frozen TTS corpus and language profile."""
    profile = load_language_profile(profile_path)
    version_path = artifacts / "dataset_version.json"
    inventory_path = artifacts / "inventory.json"
    quality_path = artifacts / "quality_report.json"
    audit_path = artifacts / "audit.jsonl"
    train_path = artifacts / "train.json"
    validation_path = artifacts / "validation.json"
    version = _read_json(version_path)
    inventory = _read_json(inventory_path)
    quality = _read_json(quality_path)
    if inventory.get("language") != profile.code:
        raise ValueError("prepared corpus language does not match training profile")
    accepted_rows, sources = _audit_summary(audit_path)
    train_rows = _count_jsonl(train_path)
    validation_rows = _count_jsonl(validation_path)
    declared_accepted = int(version.get("accepted", 0))
    if declared_accepted != accepted_rows:
        raise ValueError(
            f"dataset version/audit mismatch: accepted={declared_accepted} audit_rows={accepted_rows}"
        )
    observed = _observed_characters(inventory)
    grapheme_candidates = [
        item
        for item in observed
        if any(str(category).startswith(("L", "M")) for category in item["unicode_categories"])
    ]
    blockers = _blockers(
        profile=profile,
        accepted_rows=accepted_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        quality=quality,
    )
    reviewed = set(profile.reviewed_graphemes or ())
    observed_nonspace = {str(item["char"]) for item in observed if not item["is_whitespace"]}
    outside_reviewed = sorted(observed_nonspace - reviewed) if reviewed else []
    if profile.tokenizer_ready and outside_reviewed:
        blockers.append(
            {
                "code": "observed_outside_reviewed_inventory",
                "detail": f"Corpus contains {len(outside_reviewed)} observed characters outside the reviewed inventory.",
            }
        )
    return {
        "schema_version": 2,
        "language": profile.code,
        "profile": str(profile_path),
        "profile_sha256": file_sha256(profile_path),
        "corpus": str(artifacts),
        "dataset_version": version,
        "dataset_version_sha256": file_sha256(version_path),
        "quality_report": quality,
        "accepted_rows": accepted_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "tts_policy": {
            "sample_rate": profile.tts.sample_rate,
            "frontend": profile.tts.frontend,
            "tokenizer_reviewed": profile.tts.tokenizer_reviewed,
            "g2p_reviewed": profile.tts.g2p_reviewed,
            "reviewed_graphemes": list(profile.reviewed_graphemes) if profile.reviewed_graphemes else None,
        },
        "source_summary": sources,
        "observed_character_count": len(observed),
        "observed_characters": observed,
        "grapheme_review_candidates": grapheme_candidates,
        "candidate_inventory_is_approved": False,
        "observed_outside_reviewed_inventory": outside_reviewed,
        "blockers": blockers,
        "ready_for_production_training": not blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an evidence-driven custom-language TTS readiness report")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero while production blockers remain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_readiness_report(profile_path=args.profile, artifacts=args.artifacts)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if args.strict and not report["ready_for_production_training"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
