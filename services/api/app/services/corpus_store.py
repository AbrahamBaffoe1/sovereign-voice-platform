"""Durable corpus store for governed speech intake and two-stage transcript review."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.core.errors import ConflictError, InvalidRequestError, ResourceNotFoundError

ReviewState = Literal[
    "machine_draft",
    "reviewer_1_complete",
    "reviewer_2_complete",
    "approved",
    "rejected",
]


@dataclass(frozen=True, slots=True)
class CorpusItem:
    """Public corpus record with explicit review provenance and immutable audio identity."""

    id: str
    audio_path: str
    sha256: str
    language: str
    speaker: str
    dialect: str | None
    source_id: str
    consent_attested: bool
    state: ReviewState
    duration_seconds: float
    sample_rate: int
    machine_text: str | None
    reviewer_1_text: str | None
    reviewer_1: str | None
    reviewer_2_text: str | None
    reviewer_2: str | None
    approved_text: str | None
    parent_source_id: str | None
    segment_index: int | None
    created_at: str
    updated_at: str


class CorpusStore:
    """Own corpus persistence, deduplication, review transitions, and approved exports."""

    def __init__(self, root: Path) -> None:
        """Create storage directories and initialize SQLite using WAL for concurrent readers."""
        self.root = root
        self.audio_dir = root / "audio"
        self.db_path = root / "corpus.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open one short-lived connection so request handlers never share SQLite connections."""
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        """Create idempotent tables and indexes before the first request reaches the store."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS corpus_items (
                    id TEXT PRIMARY KEY,
                    audio_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    language TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    dialect TEXT,
                    source_id TEXT NOT NULL,
                    consent_attested INTEGER NOT NULL CHECK (consent_attested IN (0, 1)),
                    state TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    sample_rate INTEGER NOT NULL,
                    machine_text TEXT,
                    reviewer_1_text TEXT,
                    reviewer_1 TEXT,
                    reviewer_2_text TEXT,
                    reviewer_2 TEXT,
                    approved_text TEXT,
                    parent_source_id TEXT,
                    segment_index INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_corpus_language_state
                    ON corpus_items(language, state);
                CREATE INDEX IF NOT EXISTS idx_corpus_source
                    ON corpus_items(source_id);
                CREATE TABLE IF NOT EXISTS corpus_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES corpus_items(id)
                );
                """
            )

    @staticmethod
    def _now() -> str:
        """Use timezone-aware UTC timestamps for stable review ordering across machines."""
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _hash(payload: bytes) -> str:
        """Hash normalized audio bytes so the same utterance cannot enter the corpus twice."""
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> CorpusItem:
        """Convert SQLite's integer boolean and row mapping into the typed corpus record."""
        data = dict(row)
        data["consent_attested"] = bool(data["consent_attested"])
        return CorpusItem(**data)

    def _audit(
        self,
        connection: sqlite3.Connection,
        item_id: str,
        action: str,
        actor: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Append a review event in the same transaction as its state change."""
        connection.execute(
            "INSERT INTO corpus_audit(item_id, action, actor, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (item_id, action, actor, json.dumps(payload or {}, ensure_ascii=False), self._now()),
        )

    def create_item(
        self,
        *,
        wav_bytes: bytes,
        language: str,
        speaker: str,
        source_id: str,
        consent_attested: bool,
        duration_seconds: float,
        sample_rate: int,
        dialect: str | None = None,
        parent_source_id: str | None = None,
        segment_index: int | None = None,
        machine_text: str | None = None,
    ) -> CorpusItem:
        """Persist one normalized WAV utterance and initialize its machine-draft review record."""
        if not consent_attested:
            raise InvalidRequestError("corpus ingestion requires consent_attested=true")
        if not language.strip() or not speaker.strip() or not source_id.strip():
            raise InvalidRequestError("language, speaker and source_id are required")
        if duration_seconds <= 0:
            raise InvalidRequestError("duration_seconds must be positive")
        if sample_rate < 8000 or sample_rate > 48000:
            raise InvalidRequestError("sample_rate must be between 8000 and 48000")

        digest = self._hash(wav_bytes)
        item_id = uuid.uuid4().hex
        audio_path = self.audio_dir / f"{item_id}.wav"
        now = self._now()
        with self._connect() as connection:
            duplicate = connection.execute(
                "SELECT id FROM corpus_items WHERE sha256 = ?",
                (digest,),
            ).fetchone()
            if duplicate:
                raise ConflictError(f"duplicate corpus audio already exists as {duplicate['id']}")
            audio_path.write_bytes(wav_bytes)
            try:
                connection.execute(
                    """
                    INSERT INTO corpus_items(
                        id, audio_path, sha256, language, speaker, dialect, source_id,
                        consent_attested, state, duration_seconds, sample_rate, machine_text,
                        parent_source_id, segment_index, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        str(audio_path),
                        digest,
                        language.strip(),
                        speaker.strip(),
                        dialect.strip() if dialect else None,
                        source_id.strip(),
                        1,
                        "machine_draft",
                        float(duration_seconds),
                        int(sample_rate),
                        machine_text.strip() if machine_text else None,
                        parent_source_id,
                        segment_index,
                        now,
                        now,
                    ),
                )
                self._audit(connection, item_id, "ingested", "system", {"source_id": source_id})
            except Exception:
                audio_path.unlink(missing_ok=True)
                raise
        return self.get(item_id)

    def get(self, item_id: str) -> CorpusItem:
        """Load one corpus item or raise a domain-level not-found error."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM corpus_items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise ResourceNotFoundError(f"corpus item not found: {item_id}")
        return self._row_to_item(row)

    def list_items(
        self,
        *,
        language: str | None = None,
        state: ReviewState | str | None = None,
        limit: int = 200,
    ) -> list[CorpusItem]:
        """List newest records with optional language/state filters for reviewer queues."""
        if limit < 1 or limit > 1000:
            raise InvalidRequestError("limit must be between 1 and 1000")
        clauses: list[str] = []
        params: list[object] = []
        if language:
            clauses.append("language = ?")
            params.append(language)
        if state:
            clauses.append("state = ?")
            params.append(state)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM corpus_items {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def set_machine_draft(self, item_id: str, text: str, *, actor: str = "asr") -> CorpusItem:
        """Replace only the machine draft; model output never advances human review state."""
        cleaned = " ".join(text.split())
        if not cleaned:
            raise InvalidRequestError("machine transcript cannot be empty")
        with self._connect() as connection:
            row = connection.execute("SELECT state FROM corpus_items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise ResourceNotFoundError(f"corpus item not found: {item_id}")
            if row["state"] != "machine_draft":
                raise ConflictError("machine draft can only be changed before reviewer 1 completes")
            connection.execute(
                "UPDATE corpus_items SET machine_text = ?, updated_at = ? WHERE id = ?",
                (cleaned, self._now(), item_id),
            )
            self._audit(connection, item_id, "machine_draft_updated", actor)
        return self.get(item_id)

    def reviewer_1(self, item_id: str, *, reviewer: str, text: str) -> CorpusItem:
        """Record the first independent human transcript and move the item to reviewer-2 queue."""
        return self._human_transition(
            item_id,
            expected="machine_draft",
            next_state="reviewer_1_complete",
            text_column="reviewer_1_text",
            reviewer_column="reviewer_1",
            reviewer=reviewer,
            text=text,
        )

    def reviewer_2(self, item_id: str, *, reviewer: str, text: str) -> CorpusItem:
        """Record the second independent review without yet claiming the label is approved."""
        item = self.get(item_id)
        if item.reviewer_1 == reviewer:
            raise InvalidRequestError("reviewer 2 must be a different person from reviewer 1")
        return self._human_transition(
            item_id,
            expected="reviewer_1_complete",
            next_state="reviewer_2_complete",
            text_column="reviewer_2_text",
            reviewer_column="reviewer_2",
            reviewer=reviewer,
            text=text,
        )

    def _human_transition(
        self,
        item_id: str,
        *,
        expected: ReviewState,
        next_state: ReviewState,
        text_column: str,
        reviewer_column: str,
        reviewer: str,
        text: str,
    ) -> CorpusItem:
        """Apply one guarded human-review transition and audit it atomically."""
        cleaned = " ".join(text.split())
        if not reviewer.strip() or not cleaned:
            raise InvalidRequestError("reviewer and transcript text are required")
        allowed = {"reviewer_1_text", "reviewer_1", "reviewer_2_text", "reviewer_2"}
        if text_column not in allowed or reviewer_column not in allowed:
            raise RuntimeError("unsafe review column configuration")
        with self._connect() as connection:
            row = connection.execute("SELECT state FROM corpus_items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise ResourceNotFoundError(f"corpus item not found: {item_id}")
            if row["state"] != expected:
                raise ConflictError(f"expected state {expected}, found {row['state']}")
            query = (
                f"UPDATE corpus_items SET {text_column} = ?, {reviewer_column} = ?, "
                "state = ?, updated_at = ? WHERE id = ?"
            )
            connection.execute(query, (cleaned, reviewer.strip(), next_state, self._now(), item_id))
            self._audit(connection, item_id, next_state, reviewer.strip())
        return self.get(item_id)

    def approve(self, item_id: str, *, approver: str) -> CorpusItem:
        """Approve only after reviewer 2; that reviewed text becomes the training label."""
        if not approver.strip():
            raise InvalidRequestError("approver is required")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, reviewer_2_text FROM corpus_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            if row is None:
                raise ResourceNotFoundError(f"corpus item not found: {item_id}")
            if row["state"] != "reviewer_2_complete":
                raise ConflictError(f"item cannot be approved from state {row['state']}")
            if not row["reviewer_2_text"]:
                raise ConflictError("reviewer 2 transcript is missing")
            connection.execute(
                "UPDATE corpus_items SET approved_text = reviewer_2_text, state = 'approved', updated_at = ? WHERE id = ?",
                (self._now(), item_id),
            )
            self._audit(connection, item_id, "approved", approver.strip())
        return self.get(item_id)

    def reject(self, item_id: str, *, actor: str, reason: str) -> CorpusItem:
        """Reject a sample while retaining audio/provenance and the reason in the audit trail."""
        if not actor.strip() or not reason.strip():
            raise InvalidRequestError("actor and rejection reason are required")
        with self._connect() as connection:
            row = connection.execute("SELECT state FROM corpus_items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise ResourceNotFoundError(f"corpus item not found: {item_id}")
            if row["state"] in {"approved", "rejected"}:
                raise ConflictError(f"item cannot be rejected from terminal state {row['state']}")
            connection.execute(
                "UPDATE corpus_items SET state = 'rejected', updated_at = ? WHERE id = ?",
                (self._now(), item_id),
            )
            self._audit(connection, item_id, "rejected", actor.strip(), {"reason": reason.strip()})
        return self.get(item_id)

    def export_approved(self, output: Path, *, language: str) -> int:
        """Export approved labels in the strict CSV schema consumed by the training compiler."""
        rows = self.list_items(language=language, state="approved", limit=1000)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "audio",
                    "text",
                    "speaker",
                    "dialect",
                    "source_id",
                    "consent_attested",
                    "transcript_reviewed",
                ],
            )
            writer.writeheader()
            for item in rows:
                writer.writerow(
                    {
                        "audio": item.audio_path,
                        "text": item.approved_text or "",
                        "speaker": item.speaker,
                        "dialect": item.dialect or "",
                        "source_id": item.source_id,
                        "consent_attested": "true",
                        "transcript_reviewed": "true",
                    }
                )
        return len(rows)

    def audit_log(self, item_id: str) -> list[dict[str, object]]:
        """Return ordered audit events for compliance/debugging without exposing unrelated items."""
        self.get(item_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT action, actor, payload_json, created_at FROM corpus_audit WHERE item_id = ? ORDER BY id",
                (item_id,),
            ).fetchall()
        return [
            {
                "action": row["action"],
                "actor": row["actor"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def as_dict(self, item: CorpusItem) -> dict[str, object]:
        """Provide a JSON-ready representation without duplicating the dataclass field list."""
        return asdict(item)
