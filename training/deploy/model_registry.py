"""Filesystem model registry with immutable candidates and atomic environment pointers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Stage = Literal["candidate", "staging", "production", "retired"]
Task = Literal["asr", "tts"]


@dataclass(frozen=True, slots=True)
class ModelPointer:
    """Stable deployment pointer naming one immutable registered model artifact."""

    task: Task
    language: str
    stage: Stage
    model_id: str
    artifact_path: str
    updated_at: str
    previous_model_id: str | None = None


class ModelRegistry:
    """Keep model directories immutable and change deployment state by atomic pointer replacement."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry_root = root / "registry"
        self.deploy_root = root / "deployments"
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.deploy_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def model_dir(self, *, task: Task, language: str, model_id: str) -> Path:
        """Return the immutable directory allocated to one registered model candidate."""
        return self.registry_root / task / language / model_id

    def register(
        self,
        *,
        task: Task,
        language: str,
        model_id: str,
        artifact_path: Path,
        metadata: dict[str, object],
    ) -> Path:
        """Register an existing checkpoint by writing its model card exactly once."""
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_path)
        target = self.model_dir(task=task, language=language, model_id=model_id)
        if target.exists():
            raise FileExistsError(f"model already registered: {target}")
        target.mkdir(parents=True, exist_ok=False)
        card = {
            "schema_version": 1,
            "task": task,
            "language": language,
            "model_id": model_id,
            "artifact_path": str(artifact_path.resolve()),
            "registered_at": self._now(),
            "metadata": metadata,
        }
        (target / "model_card.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def pointer_path(self, *, task: Task, language: str, stage: Stage) -> Path:
        return self.deploy_root / task / language / f"{stage}.json"

    def read_pointer(self, *, task: Task, language: str, stage: Stage) -> ModelPointer | None:
        path = self.pointer_path(task=task, language=language, stage=stage)
        if not path.exists():
            return None
        return ModelPointer(**json.loads(path.read_text(encoding="utf-8")))

    def _atomic_write(self, path: Path, payload: dict[str, object]) -> None:
        """Replace a pointer atomically so readers cannot observe a partially written deployment file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)

    def promote(self, *, task: Task, language: str, model_id: str, stage: Stage) -> ModelPointer:
        """Point an environment at a registered model while retaining its rollback predecessor."""
        if stage not in {"staging", "production", "retired"}:
            raise ValueError("promotion stage must be staging, production or retired")
        card_path = self.model_dir(task=task, language=language, model_id=model_id) / "model_card.json"
        if not card_path.exists():
            raise FileNotFoundError(f"registered model card does not exist: {card_path}")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        previous = self.read_pointer(task=task, language=language, stage=stage)
        pointer = ModelPointer(
            task=task,
            language=language,
            stage=stage,
            model_id=model_id,
            artifact_path=str(card["artifact_path"]),
            updated_at=self._now(),
            previous_model_id=previous.model_id if previous else None,
        )
        self._atomic_write(self.pointer_path(task=task, language=language, stage=stage), asdict(pointer))
        return pointer

    def rollback(self, *, task: Task, language: str, stage: Stage = "production") -> ModelPointer:
        """Move a stage pointer to its predecessor without changing either model artifact."""
        current = self.read_pointer(task=task, language=language, stage=stage)
        if current is None:
            raise FileNotFoundError("deployment pointer does not exist")
        if not current.previous_model_id:
            raise ValueError("deployment pointer has no rollback predecessor")
        return self.promote(
            task=task, language=language, model_id=current.previous_model_id, stage=stage
        )
