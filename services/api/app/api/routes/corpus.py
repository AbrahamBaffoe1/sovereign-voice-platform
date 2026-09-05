"""Governed corpus-ingestion and transcript-review HTTP routes."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Form, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.errors import InvalidRequestError
from app.services.corpus_audio import normalize_clip, segment_recording

router = APIRouter(prefix="/v1/corpus", tags=["corpus"])


class ReviewRequest(BaseModel):
    """One human review action with an explicit reviewer identity and complete transcript."""

    reviewer: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20000)


class ApprovalRequest(BaseModel):
    """Final approval action; reviewer 2 already owns the approved label candidate."""

    approver: str = Field(min_length=1, max_length=200)


class RejectionRequest(BaseModel):
    """Terminal rejection action retained in the corpus audit trail."""

    actor: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)


async def _limited_upload(file: UploadFile, limit: int) -> bytes:
    """Read one byte beyond the size limit so oversized bodies are rejected without unbounded buffering."""
    payload = await file.read(limit + 1)
    if len(payload) > limit:
        raise InvalidRequestError("uploaded corpus audio exceeds configured size limit")
    if not payload:
        raise InvalidRequestError("uploaded corpus audio is empty")
    return payload


@router.post("/items")
async def create_item(
    request: Request,
    file: UploadFile,
    language: str = Form(...),
    speaker: str = Form(...),
    source_id: str = Form(...),
    consent_attested: bool = Form(...),
    dialect: str | None = Form(default=None),
) -> dict[str, object]:
    """Normalize one short utterance and place it into the machine-draft review queue."""
    container = request.app.state.container
    spec = container.languages.get(language)
    payload = await _limited_upload(file, container.settings.max_upload_bytes)
    segment = normalize_clip(
        payload,
        max_seconds=container.settings.max_corpus_clip_seconds,
        target_rate=container.settings.default_sample_rate,
    )
    item = container.corpus.create_item(
        wav_bytes=segment.wav_bytes,
        language=spec.code,
        speaker=speaker,
        source_id=source_id,
        consent_attested=consent_attested,
        duration_seconds=segment.duration_seconds,
        sample_rate=segment.sample_rate,
        dialect=dialect,
    )
    return asdict(item)


@router.post("/recordings")
async def create_recording(
    request: Request,
    file: UploadFile,
    language: str = Form(...),
    speaker: str = Form(...),
    source_id: str = Form(...),
    consent_attested: bool = Form(...),
    dialect: str | None = Form(default=None),
    multi_speaker: bool = Form(default=False),
) -> dict[str, object]:
    """Normalize and segment a long single-speaker recording; quarantine multi-speaker audio."""
    if multi_speaker:
        raise InvalidRequestError("multi-speaker recordings require diarization before corpus ingestion")
    container = request.app.state.container
    spec = container.languages.get(language)
    payload = await _limited_upload(file, container.settings.max_upload_bytes)
    segments = segment_recording(
        payload,
        max_seconds=container.settings.max_corpus_recording_seconds,
        target_rate=container.settings.default_sample_rate,
    )
    if not segments:
        raise InvalidRequestError("no speech segments were detected")
    created: list[dict[str, object]] = []
    for index, segment in enumerate(segments):
        item = container.corpus.create_item(
            wav_bytes=segment.wav_bytes,
            language=spec.code,
            speaker=speaker,
            source_id=f"{source_id}#segment-{index:04d}",
            consent_attested=consent_attested,
            duration_seconds=segment.duration_seconds,
            sample_rate=segment.sample_rate,
            dialect=dialect,
            parent_source_id=source_id,
            segment_index=index,
        )
        created.append(asdict(item))
    return {"source_id": source_id, "segments": created}


@router.get("/items")
async def list_items(
    request: Request,
    language: str | None = None,
    state: str | None = None,
    limit: int = 200,
) -> list[dict[str, object]]:
    """Return the reviewer queue while preserving every item's explicit state."""
    canonical = request.app.state.container.languages.get(language).code if language else None
    allowed = {"machine_draft", "reviewer_1_complete", "reviewer_2_complete", "approved", "rejected"}
    if state is not None and state not in allowed:
        raise InvalidRequestError(f"unknown corpus review state: {state}")
    items = request.app.state.container.corpus.list_items(language=canonical, state=state, limit=limit)
    return [asdict(item) for item in items]


@router.get("/items/{item_id}")
async def get_item(item_id: str, request: Request) -> dict[str, object]:
    """Return one corpus item and its append-only audit history."""
    store = request.app.state.container.corpus
    return {"item": asdict(store.get(item_id)), "audit": store.audit_log(item_id)}


@router.post("/items/{item_id}/review-1")
async def review_1(item_id: str, body: ReviewRequest, request: Request) -> dict[str, object]:
    """Apply reviewer 1's full transcript correction."""
    return asdict(request.app.state.container.corpus.reviewer_1(item_id, reviewer=body.reviewer, text=body.text))


@router.post("/items/{item_id}/review-2")
async def review_2(item_id: str, body: ReviewRequest, request: Request) -> dict[str, object]:
    """Apply the independent reviewer 2 transcript."""
    return asdict(request.app.state.container.corpus.reviewer_2(item_id, reviewer=body.reviewer, text=body.text))


@router.post("/items/{item_id}/approve")
async def approve(item_id: str, body: ApprovalRequest, request: Request) -> dict[str, object]:
    """Promote the reviewer-2 transcript to the immutable approved label."""
    return asdict(request.app.state.container.corpus.approve(item_id, approver=body.approver))


@router.post("/items/{item_id}/reject")
async def reject(item_id: str, body: RejectionRequest, request: Request) -> dict[str, object]:
    """Reject a corpus item while retaining its provenance and reason."""
    return asdict(request.app.state.container.corpus.reject(item_id, actor=body.actor, reason=body.reason))
