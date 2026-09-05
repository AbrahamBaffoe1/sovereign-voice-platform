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
    """Canonical speech sample used during dataset preparation before serialization into NeMo or
    generic JSONL formats."""
    audio_filepath: str
    text: str
    duration: float
    speaker: str | None = None
    language: str | None = None
    sha256: str | None = None

    def nemo_dict(self) -> dict[str, object]:
        """Project a SpeechRecord into the fields consumed by NeMo while omitting internal audit
        metadata such as the source SHA-256."""
        payload: dict[str, object] = {"audio_filepath": self.audio_filepath,"text": self.text,"duration": round(self.duration, 5)}
        if self.speaker is not None: payload["speaker"] = self.speaker
        if self.language is not None: payload["language"] = self.language
        return payload


def normalize_transcript(text: str) -> str:
    """Apply only Unicode NFC and whitespace canonicalization so dataset preparation remains
    language-neutral and does not invent pronunciation transformations."""
    return " ".join(unicodedata.normalize("NFC", text).replace("\u00a0", " ").split())


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an audio file incrementally so large corpora can be deduplicated without loading whole
    recordings into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size): digest.update(chunk)
    return digest.hexdigest()


def stable_partition(key: str, train: float = 0.90, validation: float = 0.05) -> str:
    """Map a stable key into train/validation/test buckets using SHA-256. Hash-based splits remain
    unchanged when CSV row order changes or the compiler is rerun."""
    if train <= 0 or validation < 0 or train + validation >= 1: raise ValueError("split fractions must leave a non-zero test partition")
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train: return "train"
    if bucket < train + validation: return "validation"
    return "test"


def write_jsonl(path: Path, records: Iterable[SpeechRecord], *, nemo: bool = True) -> None:
    """Write speech records one JSON object per line, creating parent directories and preserving
    Unicode text for corpus inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.nemo_dict() if nemo else asdict(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read and validate a JSONL manifest with precise file/line errors so corrupt training inputs fail
    before expensive GPU jobs start."""
    rows=[]
    with path.open("r", encoding="utf-8") as handle:
        for line_no,line in enumerate(handle,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row,dict): raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows
