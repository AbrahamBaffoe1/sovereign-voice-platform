"""Generate ASR machine drafts without advancing human review state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.corpus_store import CorpusStore


def draft_pending(
    *,
    corpus_root: Path,
    model: str,
    device: str = "auto",
    compute_type: str = "default",
    language: str | None = None,
    limit: int = 500,
) -> dict[str, object]:
    """Transcribe machine-draft items and store model output as non-authoritative suggestions."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("install the 'asr' extra to generate machine drafts") from exc
    store = CorpusStore(corpus_root)
    rows = store.list_items(language=language, state="machine_draft", limit=limit)
    whisper = WhisperModel(model, device=device, compute_type=compute_type)
    drafted = 0
    failures: list[dict[str, str]] = []
    for item in rows:
        try:
            segments, _ = whisper.transcribe(
                item.audio_path, language=None, vad_filter=True, beam_size=5
            )
            text = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            ).strip()
            if not text:
                failures.append({"item_id": item.id, "error": "empty_transcript"})
                continue
            store.set_machine_draft(item.id, text, actor=f"faster-whisper:{model}")
            drafted += 1
        except Exception as exc:
            failures.append({"item_id": item.id, "error": f"{type(exc).__name__}:{exc}"})
    return {"drafted": drafted, "failures": failures, "examined": len(rows)}


def main() -> None:
    """Generate drafts from the command line while keeping model/runtime knobs explicit."""
    parser = argparse.ArgumentParser(description="Generate machine transcript drafts for corpus review")
    parser.add_argument("--corpus-root", type=Path, default=Path("data/corpus"))
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--language", default=None)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    print(
        json.dumps(
            draft_pending(
                corpus_root=args.corpus_root,
                model=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                limit=args.limit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
