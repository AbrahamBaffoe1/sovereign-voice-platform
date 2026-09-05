"""Provider-schema adapters mapping external rows into governed speech metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptedRow:
    """Minimal normalized metadata produced before provider audio enters a corpus workspace."""

    audio: str
    text: str
    speaker: str | None
    dialect: str | None
    source_id: str
    training_only: bool = False


def _required(row: Mapping[str, object], *names: str) -> object:
    """Return the first non-empty provider field from a reviewed alias list."""
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    raise ValueError(f"missing required provider field; tried {names}")


def adapt_waxal(row: Mapping[str, object], *, source_prefix: str = "waxal") -> AdaptedRow:
    """Map a WAXAL-style row while retaining real speaker identity for leakage-safe splits."""
    audio = str(_required(row, "audio", "path", "audio_filepath"))
    text = str(_required(row, "text", "transcription", "sentence"))
    speaker = str(_required(row, "speaker_id", "speaker", "client_id"))
    row_id = str(row.get("id") or row.get("utt_id") or audio)
    dialect = str(row["dialect"]).strip() if row.get("dialect") else None
    return AdaptedRow(audio, text, speaker, dialect, f"{source_prefix}:{row_id}")


def adapt_ga_parallel(row: Mapping[str, object], *, source_prefix: str = "ga-parallel") -> AdaptedRow:
    """Map Ga data without fabricating speaker IDs; unknown-speaker rows are training-only."""
    audio = str(_required(row, "audio", "path", "audio_filepath"))
    text = str(_required(row, "text", "sentence", "transcription", "ga"))
    row_id = str(row.get("id") or row.get("utt_id") or audio)
    return AdaptedRow(audio, text, None, None, f"{source_prefix}:{row_id}", training_only=True)


def adapt_common_voice(row: Mapping[str, object], *, source_prefix: str = "common-voice") -> AdaptedRow:
    """Map Common Voice rows and use client_id as speaker identity when available."""
    audio = str(_required(row, "audio", "path"))
    text = str(_required(row, "sentence", "text"))
    speaker = str(row["client_id"]).strip() if row.get("client_id") else None
    row_id = str(row.get("id") or row.get("path") or audio)
    dialect = str(row["accent"]).strip() if row.get("accent") else None
    return AdaptedRow(audio, text, speaker, dialect, f"{source_prefix}:{row_id}", training_only=speaker is None)
