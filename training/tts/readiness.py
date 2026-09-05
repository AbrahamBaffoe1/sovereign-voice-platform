"""Aggregate evidence-driven TTS readiness reports across frozen corpus-v0 languages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.tts.preflight import build_readiness_report

_LANGUAGES = ("tw", "gaa", "ee", "ha")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS readiness review packets for corpus-v0")
    parser.add_argument("--language", choices=[*_LANGUAGES, "all"], default="all")
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/bootstrap"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("training/configs/languages"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/tts-readiness"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero while any requested language has blockers.")
    args = parser.parse_args()
    languages = list(_LANGUAGES) if args.language == "all" else [args.language]
    reports = [
        build_readiness_report(
            profile_path=args.profiles_dir / f"{language}.yaml",
            artifacts=args.artifacts_root / language / "tts" / "corpus-v0",
        )
        for language in languages
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for report in reports:
        (args.output_root / f"{report['language']}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    result = {
        "status": "ready" if all(report["ready_for_production_training"] for report in reports) else "blocked",
        "languages": [
            {
                "language": report["language"],
                "ready": report["ready_for_production_training"],
                "blocker_codes": [item["code"] for item in report["blockers"]],
                "report": str(args.output_root / f"{report['language']}.json"),
            }
            for report in reports
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.strict and result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
