"""Shared speech-manifest primitives: normalization, hashing, deterministic splitting, and JSONL I/O."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SpeechRecord:
    """Canonical speech sample used during preparation before NeMo/HF serialization."""
    audio_filepath: str
    text: str
    duration: float
    speaker: str | None = None
    language: str | None = None
    sha256: str | None = None
    dialect: str | None = None
    source_id: str | None = None
    consent_attested: bool | None = None
    transcript_reviewed: bool | None = None
    split: str | None = None

    def nemo_dict(self) -> dict[str, object]:
        """Project a record into the fields consumed by NeMo, excluding audit-only metadata."""
        payload: dict[str, object] = {
            "audio_filepath": self.audio_filepath,
            "text": self.text,
            "duration": round(self.duration, 5),
        }
        if self.speaker is not None:
            payload["speaker"] = self.speaker
        if self.language is not None:
            payload["language"] = self.language
        return payload


def normalize_transcript(text: str) -> str:
    """Apply only Unicode NFC and whitespace canonicalization; never invent spoken-language rules."""
    return " ".join(unicodedata.normalize("NFC", text).replace("\u00a0", " ").split())


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash audio incrementally so large corpora can be deduplicated without loading files into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_partition(key: str, train: float = 0.90, validation: float = 0.05) -> str:
    """Map a stable speaker/audio key into deterministic train/validation/test buckets."""
    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("split fractions must leave a non-zero test partition")
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def dataset_fingerprint(records: Iterable[SpeechRecord]) -> str:
    """Hash accepted record identity/text/provenance into a stable dataset version fingerprint."""
    digest = hashlib.sha256()
    canonical = sorted(
        (
            record.sha256 or "",
            normalize_transcript(record.text),
            record.speaker or "",
            record.dialect or "",
            record.source_id or "",
            record.split or "",
        )
        for record in records
    )
    for row in canonical:
        digest.update("\x1f".join(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[SpeechRecord], *, nemo: bool = True) -> None:
    """Write records one JSON object per line while preserving Unicode for human corpus inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.nemo_dict() if nemo else asdict(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL manifest with precise line errors so bad inputs fail before expensive GPU jobs."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows
