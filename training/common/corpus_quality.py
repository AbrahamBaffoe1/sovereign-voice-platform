"""Corpus-level quality summaries used to detect leakage and underrepresented speakers/dialects."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from training.common.manifest import SpeechRecord


@dataclass(frozen=True, slots=True)
class CorpusQualityReport:
    """Compact, JSON-serializable aggregate of a prepared multilingual speech corpus."""
    accepted: int
    hours: float
    speakers: int
    dialects: dict[str, int]
    split_rows: dict[str, int]
    split_hours: dict[str, float]
    speaker_leakage: list[str]

    def as_dict(self) -> dict[str, object]:
        """Convert the report to primitives suitable for JSON artifacts and CI output."""
        return {
            "accepted": self.accepted,
            "hours": round(self.hours, 4),
            "speakers": self.speakers,
            "dialects": self.dialects,
            "split_rows": self.split_rows,
            "split_hours": {key: round(value, 4) for key, value in self.split_hours.items()},
            "speaker_leakage": self.speaker_leakage,
        }


def build_quality_report(records: list[SpeechRecord]) -> CorpusQualityReport:
    """Aggregate duration, dialect coverage and speaker split leakage from accepted records."""
    dialects: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_seconds: defaultdict[str, float] = defaultdict(float)
    speaker_splits: defaultdict[str, set[str]] = defaultdict(set)
    speakers: set[str] = set()
    for record in records:
        split = record.split or "unknown"
        split_rows[split] += 1
        split_seconds[split] += record.duration
        if record.dialect:
            dialects[record.dialect] += 1
        if record.speaker:
            speakers.add(record.speaker)
            speaker_splits[record.speaker].add(split)
    leakage = sorted(speaker for speaker, splits in speaker_splits.items() if len(splits) > 1)
    return CorpusQualityReport(
        accepted=len(records),
        hours=sum(item.duration for item in records) / 3600.0,
        speakers=len(speakers),
        dialects=dict(dialects),
        split_rows=dict(split_rows),
        split_hours={key: seconds / 3600.0 for key, seconds in split_seconds.items()},
        speaker_leakage=leakage,
    )
