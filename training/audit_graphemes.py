"""Pre-training gate that fails when corpus text contains graphemes missing from the reviewed language alphabet."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from training.common.manifest import read_jsonl


def main() -> None:
    """Compare every character observed in a prepared manifest against the reviewed alphabet. Unknown
    graphemes are reported with counts/codepoints and cause a non-zero exit so training cannot
    silently discard symbols."""
    parser = argparse.ArgumentParser(description="Audit grapheme coverage before TTS training")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language-config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.language_config.read_text(encoding="utf-8")) or {}
    alphabet = config.get("alphabet")
    if not isinstance(alphabet, str) or not alphabet:
        raise SystemExit("language alphabet is empty; populate it from the observed corpus inventory and native review first")
    allowed = set(alphabet)
    counts: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    for row in read_jsonl(args.manifest):
        text = str(row.get("text", ""))
        counts.update(text)
        unknown.update(ch for ch in text if ch not in allowed)
    report = {"unique_graphemes":len(counts),"unknown_unique":len(unknown),"unknown":[{"char":ch,"count":n,"codepoint":f"U+{ord(ch):04X}"} for ch,n in unknown.most_common()]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if unknown:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
