"""Deterministic ASR benchmark metrics with diagnostic slices and serving-latency statistics."""

from __future__ import annotations

import json
import math
import statistics
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


def normalize_for_error_rate(text: str) -> str:
    """Apply Unicode/whitespace normalization without deleting meaningful graphemes."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Compute Levenshtein distance with one rolling row instead of O(n*m) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for index, ref_token in enumerate(reference, 1):
        current = [index]
        for column, hyp_token in enumerate(hypothesis, 1):
            substitution = previous[column - 1] + (ref_token != hyp_token)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, unit: str) -> tuple[int, int]:
    """Return raw edit count and reference-token count for exact corpus aggregation."""
    ref = normalize_for_error_rate(reference)
    hyp = normalize_for_error_rate(hypothesis)
    if unit == "word":
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
    elif unit == "char":
        ref_tokens = list(ref.replace(" ", ""))
        hyp_tokens = list(hyp.replace(" ", ""))
    else:
        raise ValueError("unit must be 'word' or 'char'")
    return _edit_distance(ref_tokens, hyp_tokens), len(ref_tokens)


@dataclass(frozen=True, slots=True)
class BenchmarkUtterance:
    """One held-out ASR result plus metadata needed to expose subgroup failures."""

    reference: str
    hypothesis: str
    audio_seconds: float
    latency_seconds: float
    speaker: str | None = None
    dialect: str | None = None
    noise: str | None = None
    code_switch: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Aggregated WER/CER counts with reproducible numerators and denominators."""

    utterances: int
    word_errors: int
    reference_words: int
    char_errors: int
    reference_chars: int
    wer: float
    cer: float


def _summary(rows: Iterable[BenchmarkUtterance]) -> ErrorSummary:
    count = word_errors = reference_words = char_errors = reference_chars = 0
    for row in rows:
        count += 1
        edits, total = error_rate(row.reference, row.hypothesis, unit="word")
        word_errors += edits
        reference_words += total
        edits, total = error_rate(row.reference, row.hypothesis, unit="char")
        char_errors += edits
        reference_chars += total
    return ErrorSummary(
        utterances=count,
        word_errors=word_errors,
        reference_words=reference_words,
        char_errors=char_errors,
        reference_chars=reference_chars,
        wer=word_errors / reference_words if reference_words else math.nan,
        cer=char_errors / reference_chars if reference_chars else math.nan,
    )


def _percentile(values: list[float], percentile: float) -> float:
    """Compute a linearly interpolated percentile without a dataframe dependency."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def benchmark_report(rows: list[BenchmarkUtterance]) -> dict[str, object]:
    """Produce global accuracy, latency/RTF, and speaker/dialect/noise/code-switch slices."""
    global_summary = _summary(rows)
    latencies = [row.latency_seconds for row in rows]
    audio_seconds = sum(max(row.audio_seconds, 0.0) for row in rows)
    total_latency = sum(max(row.latency_seconds, 0.0) for row in rows)
    report: dict[str, object] = {
        "global": asdict(global_summary),
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else math.nan,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "real_time_factor": total_latency / audio_seconds if audio_seconds else math.nan,
        "slices": {},
    }
    slices: dict[str, dict[str, dict[str, object]]] = {}
    for attribute in ("speaker", "dialect", "noise", "code_switch"):
        grouped: dict[str, list[BenchmarkUtterance]] = defaultdict(list)
        for row in rows:
            value = getattr(row, attribute)
            grouped[str(value) if value else "unknown"].append(row)
        slices[attribute] = {key: asdict(_summary(group)) for key, group in sorted(grouped.items())}
    report["slices"] = slices
    return report


def rows_from_jsonl(path: Path) -> list[BenchmarkUtterance]:
    """Load benchmark rows from JSONL with strict required-field validation."""
    rows: list[BenchmarkUtterance] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            try:
                rows.append(
                    BenchmarkUtterance(
                        reference=str(payload["reference"]),
                        hypothesis=str(payload["hypothesis"]),
                        audio_seconds=float(payload["audio_seconds"]),
                        latency_seconds=float(payload["latency_seconds"]),
                        speaker=str(payload["speaker"]) if payload.get("speaker") else None,
                        dialect=str(payload["dialect"]) if payload.get("dialect") else None,
                        noise=str(payload["noise"]) if payload.get("noise") else None,
                        code_switch=str(payload["code_switch"]) if payload.get("code_switch") else None,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: malformed benchmark row") from exc
    return rows
