"""Offline ASR evaluation tool that reports WER/CER and preserves per-sample hypotheses for error analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Load a deployable Faster-Whisper checkpoint, transcribe every manifest sample with fixed
    decoding settings, compute corpus WER/CER, and emit per-sample reference/hypothesis pairs for
    human error analysis."""
    parser = argparse.ArgumentParser(description="Evaluate a Faster-Whisper/CTranslate2 checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
        from jiwer import cer, wer
    except ImportError as exc:
        raise SystemExit("Install faster-whisper and jiwer before evaluation") from exc

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict[str, object]] = []

    with args.manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            reference = " ".join(str(sample["text"]).split())
            segments, info = model.transcribe(
                str(sample["audio_filepath"]),
                language=args.language,
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=5,
            )
            hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
            references.append(reference)
            hypotheses.append(hypothesis)
            rows.append(
                {
                    "index": index,
                    "audio": sample["audio_filepath"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "detected_language": info.language,
                }
            )

    result = {
        "samples": len(rows),
        "wer": wer(references, hypotheses) if rows else None,
        "cer": cer(references, hypotheses) if rows else None,
        "errors": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
