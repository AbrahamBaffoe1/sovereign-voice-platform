"""Version-controlled registry for ASR teachers used in benchmark and distillation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TeacherSpec:
    """Resolved teacher checkpoint metadata with an explicit access and family policy."""

    name: str
    model_id: str
    family: str
    language: str
    access: str
    role: str
    license: str | None = None
    blocked_reason: str | None = None
    notes: str | None = None
    language_prefix_id: int | None = None

    @property
    def runnable(self) -> bool:
        """Return whether the repository considers this teacher directly executable."""
        return self.access == "public" and not self.blocked_reason


def load_teacher_registry(path: Path = Path("training/configs/asr_teachers.yaml")) -> dict[str, TeacherSpec]:
    """Load and validate teacher metadata without importing heavyweight model dependencies."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported teacher registry schema: {path}")
    raw_teachers = payload.get("teachers")
    if not isinstance(raw_teachers, dict) or not raw_teachers:
        raise ValueError(f"teacher registry is empty: {path}")
    resolved: dict[str, TeacherSpec] = {}
    for name, raw in raw_teachers.items():
        if not isinstance(raw, dict):
            raise ValueError(f"teacher {name!r} must be a mapping")
        required = ("model_id", "family", "language", "access", "role")
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"teacher {name!r} missing fields: {', '.join(missing)}")
        family = str(raw["family"])
        if family not in {"ctc", "seq2seq"}:
            raise ValueError(f"teacher {name!r} has unsupported family: {family}")
        access = str(raw["access"])
        if access not in {"public", "gated"}:
            raise ValueError(f"teacher {name!r} has unsupported access policy: {access}")
        resolved[name] = TeacherSpec(
            name=name,
            model_id=str(raw["model_id"]),
            family=family,
            language=str(raw["language"]),
            access=access,
            role=str(raw["role"]),
            license=_optional(raw, "license"),
            blocked_reason=_optional(raw, "blocked_reason"),
            notes=_optional(raw, "notes"),
            language_prefix_id=_optional_int(raw, "language_prefix_id"),
        )
    return resolved


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """Return an optional non-negative integer field from the registry."""
    value = payload.get(key)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative")
    return parsed


def _optional(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value).strip() if value is not None and str(value).strip() else None


def get_teacher(name: str, path: Path = Path("training/configs/asr_teachers.yaml")) -> TeacherSpec:
    """Resolve one teacher and fail with a useful list of valid names."""
    teachers = load_teacher_registry(path)
    try:
        return teachers[name]
    except KeyError as exc:
        raise KeyError(f"unknown teacher {name!r}; choose from {', '.join(sorted(teachers))}") from exc
