"""License-aware speech-source catalog used before any dataset is downloaded or mixed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Usage = Literal["production", "evaluation", "research"]
Task = Literal["asr", "tts"]


@dataclass(frozen=True, slots=True)
class DataSource:
    """One governed source, including exact loader coordinates and maximum permitted usage."""

    source_id: str
    provider: str
    languages: tuple[str, ...]
    tasks: tuple[Task, ...]
    usage: Usage
    license: str
    requires_revision_pin: bool
    repo_id: str | None = None
    revision: str | None = None
    config: str | None = None
    split: str = "train"
    fields: dict[str, str] = field(default_factory=dict)
    governance_approved: bool = False
    upstream_validated: bool = False
    training_only: bool = False
    gated: bool = False
    optional: bool = False
    notes: str = ""

    def allows(self, requested: Usage) -> bool:
        """Enforce monotonic usage boundaries: production is strictest, research least restrictive."""
        rank: dict[Usage, int] = {"production": 0, "evaluation": 1, "research": 2}
        return rank[requested] >= rank[self.usage]

    def supports(self, task: Task) -> bool:
        """Return whether this source is explicitly approved for the requested model task."""
        return task in self.tasks


class SourceCatalog:
    """Load source policy once so acquisition, compilation, and release use the same decisions."""

    def __init__(self, path: Path) -> None:
        """Parse a version-controlled catalog and reject malformed governance definitions early."""
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
            raw_tasks = tuple(str(item).strip() for item in raw.get("tasks", []) if str(item).strip())
            if not languages or not raw_tasks:
                raise ValueError(f"{path}: source {source_id!r} requires languages and tasks")
            if any(task not in {"asr", "tts"} for task in raw_tasks):
                raise ValueError(f"{path}: source {source_id!r} has an invalid task")
            fields = raw.get("fields") or {}
            if not isinstance(fields, dict):
                raise ValueError(f"{path}: source {source_id!r}.fields must be a mapping")
            self.sources[str(source_id)] = DataSource(
                source_id=str(source_id),
                provider=str(raw.get("provider", "")).strip(),
                repo_id=str(raw["repo_id"]).strip() if raw.get("repo_id") else None,
                revision=str(raw["revision"]).strip() if raw.get("revision") else None,
                config=str(raw["config"]).strip() if raw.get("config") else None,
                split=str(raw.get("split", "train")).strip(),
                languages=languages,
                tasks=raw_tasks,  # type: ignore[arg-type]
                usage=usage,  # type: ignore[arg-type]
                license=str(raw.get("license", "unknown")).strip(),
                requires_revision_pin=bool(raw.get("requires_revision_pin", True)),
                fields={str(key): str(value) for key, value in fields.items() if value is not None},
                governance_approved=bool(raw.get("governance_approved", False)),
                upstream_validated=bool(raw.get("upstream_validated", False)),
                training_only=bool(raw.get("training_only", False)),
                gated=bool(raw.get("gated", False)),
                optional=bool(raw.get("optional", False)),
                notes=str(raw.get("notes", "")).strip(),
            )

    def get(self, source_id: str) -> DataSource:
        """Resolve one source by stable ID rather than duplicating policy strings across CLIs."""
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown data source: {source_id}") from exc

    def plan(self, *, language: str, usage: Usage, task: Task | None = None) -> list[DataSource]:
        """Return sources whose language, task, and usage boundaries permit an experiment."""
        return [
            source
            for source in self.sources.values()
            if language in source.languages
            and source.allows(usage)
            and (task is None or source.supports(task))
        ]
