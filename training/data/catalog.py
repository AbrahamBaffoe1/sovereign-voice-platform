"""License-aware external-dataset catalog used before data is downloaded or mixed into training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Usage = Literal["production", "evaluation", "research"]


@dataclass(frozen=True, slots=True)
class DataSource:
    """One governed data source with an explicit maximum usage scope."""

    source_id: str
    provider: str
    languages: tuple[str, ...]
    usage: Usage
    license: str
    requires_revision_pin: bool
    repo_id: str | None = None
    notes: str = ""

    def allows(self, requested: Usage) -> bool:
        """Enforce a monotonic boundary: production is strictest and research least restrictive."""
        rank: dict[Usage, int] = {"production": 0, "evaluation": 1, "research": 2}
        return rank[requested] >= rank[self.usage]


class SourceCatalog:
    """Load source policy once so every downloader and adapter shares the same decision boundary."""

    def __init__(self, path: Path) -> None:
        """Parse a version-controlled catalog and reject malformed definitions immediately."""
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, dict):
            raise ValueError(f"{path}: sources must be a mapping")
        self.path = path
        self.sources: dict[str, DataSource] = {}
        for source_id, raw in raw_sources.items():
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: source {source_id!r} must be a mapping")
            usage = str(raw.get("usage", "")).strip()
            if usage not in {"production", "evaluation", "research"}:
                raise ValueError(f"{path}: invalid usage for {source_id}: {usage!r}")
            languages = tuple(str(item).strip() for item in raw.get("languages", []) if str(item).strip())
            if not languages:
                raise ValueError(f"{path}: source {source_id!r} has no languages")
            self.sources[str(source_id)] = DataSource(
                source_id=str(source_id),
                provider=str(raw.get("provider", "")).strip(),
                repo_id=str(raw["repo_id"]).strip() if raw.get("repo_id") else None,
                languages=languages,
                usage=usage,  # type: ignore[arg-type]
                license=str(raw.get("license", "unknown")).strip(),
                requires_revision_pin=bool(raw.get("requires_revision_pin", True)),
                notes=str(raw.get("notes", "")).strip(),
            )

    def get(self, source_id: str) -> DataSource:
        """Resolve one source by stable ID rather than duplicating policy strings in CLIs."""
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown data source: {source_id}") from exc

    def plan(self, *, language: str, usage: Usage) -> list[DataSource]:
        """Return sources whose language and usage boundaries permit the requested experiment."""
        return [
            source
            for source in self.sources.values()
            if language in source.languages and source.allows(usage)
        ]
