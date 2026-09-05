"""Model-card and artifact-integrity primitives for reproducible ASR/TTS checkpoint registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ModelTask = Literal["asr", "tts-acoustic", "tts-vocoder"]


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Immutable lineage record linking one model candidate to language, data, metrics and artifacts."""
    schema_version: int
    model_id: str
    language: str
    task: ModelTask
    status: Literal["candidate", "staging", "production", "retired"]
    created_at: str
    dataset_id: str
    dataset_fingerprint_sha256: str
    base_model: str | None
    checkpoint_path: str
    artifact_manifest_sha256: str
    metrics: dict[str, object]

    def to_json(self) -> str:
        """Serialize with stable formatting so reviews and diffs remain readable."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def hash_checkpoint_tree(checkpoint: Path) -> tuple[str, list[dict[str, object]]]:
    """Hash model files and then hash the ordered file manifest as the checkpoint artifact ID."""
    if not checkpoint.exists() or not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    files: list[dict[str, object]] = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        if path.name == "model_card.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        files.append({
            "path": str(path.relative_to(checkpoint)),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    if not files:
        raise ValueError(f"checkpoint directory contains no model files: {checkpoint}")
    manifest_bytes = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(manifest_bytes).hexdigest(), files


def build_model_id(language: str, task: ModelTask, artifact_hash: str) -> str:
    """Create a readable immutable model ID from canonical language/task plus artifact hash."""
    return f"{language}-{task}-{artifact_hash[:12]}"


def utc_now() -> str:
    """Return an RFC3339-compatible UTC timestamp without relying on local machine timezone."""
    return datetime.now(UTC).isoformat()
