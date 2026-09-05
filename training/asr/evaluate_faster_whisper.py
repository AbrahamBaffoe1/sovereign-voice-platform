"""Offline Faster-Whisper evaluation with global and dialect-segmented WER/CER reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _metric_block(rows: list[dict[str, Any]], wer_fn: Any, cer_fn: Any) -> dict[str, object]:
    """Compute WER/CER for one slice while preserving a zero-sample representation."""
    if not rows:
        return {"samples": 0, "wer": None, "cer": None}
    references = [str(row["reference"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    return {
        "samples": len(rows),
        "wer": float(wer_fn(references, hypotheses)),
        "cer": float(cer_fn(references, hypotheses)),
    }


def main() -> None:
    """Transcribe held-out samples and emit overall plus dialect/speaker-group error metrics."""
    parser = argparse.ArgumentParser(description="Evaluate a Faster-Whisper/CTranslate2 checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="NeMo test manifest or rich audit.jsonl; audit rows enable dialect slices.",
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--split",
        default="test",
        help="When reading audit.jsonl, evaluate only this split. Set empty string to use all rows.",
    )
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
        from jiwer import cer, wer
    except ImportError as exc:
        raise SystemExit("Install faster-whisper and jiwer before evaluation") from exc

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    rows: list[dict[str, object]] = []
    by_dialect: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    with args.manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if args.split and sample.get("split") and sample.get("split") != args.split:
                continue
            reference = " ".join(str(sample["text"]).split())
            segments, info = model.transcribe(
                str(sample["audio_filepath"]),
                language=args.language,
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
            )
            hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
            row: dict[str, object] = {
                "index": index,
                "audio": sample["audio_filepath"],
                "speaker": sample.get("speaker"),
                "dialect": sample.get("dialect"),
                "reference": reference,
                "hypothesis": hypothesis,
                "detected_language": info.language,
            }
            rows.append(row)
            dialect = str(sample.get("dialect") or "unlabeled")
            by_dialect[dialect].append(row)

    result = {
        "model": args.model,
        "language_hint": args.language,
        "overall": _metric_block(rows, wer, cer),
        "by_dialect": {
            dialect: _metric_block(group, wer, cer) for dialect, group in sorted(by_dialect.items())
        },
        "errors": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
