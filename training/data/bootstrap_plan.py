"""Version-controlled bootstrap-corpus plans tying each language/task to approved source roles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Task = Literal["asr", "tts"]
Role = Literal["train", "eval"]


@dataclass(frozen=True, slots=True)
class PlannedSource:
    """One source ID selected for a language, model task, and non-overlapping corpus role."""

    language: str
    task: Task
    role: Role
    source_id: str


class BootstrapPlan:
    """Load the source mix without embedding dataset choices in acquisition code."""

    def __init__(self, path: Path) -> None:
        """Parse the plan once and reject malformed language/task/role structures early."""
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_languages = payload.get("languages")
        if not isinstance(raw_languages, dict):
            raise ValueError(f"{path}: languages must be a mapping")
        self.path = path
        self._languages = raw_languages

    def sources(self, *, language: str, task: Task, role: Role = "train") -> tuple[str, ...]:
        """Return source IDs in deliberate order so bootstrap runs are reproducible and reviewable."""
        raw_language = self._languages.get(language)
        if not isinstance(raw_language, dict):
            raise KeyError(f"bootstrap plan has no language {language!r}")
        raw_task = raw_language.get(task)
        if not isinstance(raw_task, dict):
            raise KeyError(f"bootstrap plan has no {task!r} task for {language!r}")
        raw_sources = raw_task.get(role, [])
        if not isinstance(raw_sources, list):
            raise ValueError(f"{self.path}: {language}.{task}.{role} must be a list")
        return tuple(str(item).strip() for item in raw_sources if str(item).strip())

    def planned(self, *, language: str, task: Task, include_eval: bool = False) -> list[PlannedSource]:
        """Expand one language/task request into training and optionally independent evaluation sources."""
        result = [
            PlannedSource(language, task, "train", source_id)
            for source_id in self.sources(language=language, task=task, role="train")
        ]
        if include_eval:
            result.extend(
                PlannedSource(language, task, "eval", source_id)
                for source_id in self.sources(language=language, task=task, role="eval")
            )
        return result
