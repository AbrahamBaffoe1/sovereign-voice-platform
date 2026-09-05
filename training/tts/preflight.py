"""TTS training preflight that blocks unreviewed tokenizers and corpus-policy violations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common.language_profile import load_language_profile
from training.common.manifest import read_jsonl


def parse_args() -> argparse.Namespace:
    """Accept one reviewed language profile and prepared artifact directory for deterministic checks."""
    parser = argparse.ArgumentParser(description="Validate a custom-language corpus before TTS training")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--allow-unreviewed-tokenizer", action="store_true", help="Research-only escape hatch; production training should never use this.")
    return parser.parse_args()


def main() -> None:
    """Fail fast on missing manifests, speaker leakage, unreviewed graphemes or tokenizer policy."""
    args = parse_args(); profile = load_language_profile(args.profile)
    inventory_path = args.artifacts / "inventory.json"; quality_path = args.artifacts / "quality_report.json"
    for path in (inventory_path, quality_path, args.artifacts / "train.json"):
        if not path.exists(): raise SystemExit(f"missing required prepared artifact: {path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")); quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if inventory.get("language") != profile.code: raise SystemExit("prepared corpus language does not match training profile")
    if quality.get("speaker_leakage"): raise SystemExit(f"speaker leakage detected: {quality['speaker_leakage']}")
    observed = {item["char"] for item in inventory.get("characters", []) if not item["char"].isspace()}
    if not args.allow_unreviewed_tokenizer:
        if not profile.tokenizer_ready:
            raise SystemExit("TTS tokenizer is not approved: set tokenizer_reviewed=true and reviewed_graphemes only after native-speaker/linguist review")
        reviewed = set(profile.reviewed_graphemes or ()); unknown = sorted(observed - reviewed)
        if unknown: raise SystemExit(f"corpus contains graphemes outside reviewed inventory: {unknown}")
    rows = read_jsonl(args.artifacts / "train.json")
    if not rows: raise SystemExit("training manifest is empty")
    print(json.dumps({"language": profile.code, "rows": len(rows), "hours": quality.get("hours"), "speakers": quality.get("speakers"), "tokenizer_ready": profile.tokenizer_ready, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
