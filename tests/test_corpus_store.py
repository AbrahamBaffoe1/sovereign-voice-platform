"""Review-workflow tests for the durable speech corpus store."""

from pathlib import Path

import pytest

from app.core.errors import ConflictError, InvalidRequestError
from app.services.corpus_store import CorpusStore


def _item(store: CorpusStore):
    return store.create_item(
        wav_bytes=b"RIFF-test-audio",
        language="tw",
        speaker="speaker-1",
        source_id="call-1#segment-0000",
        consent_attested=True,
        duration_seconds=1.0,
        sample_rate=16000,
    )


def test_review_requires_two_different_reviewers(tmp_path: Path) -> None:
    """One person must not satisfy both transcript-review stages."""
    store = CorpusStore(tmp_path)
    item = _item(store)
    store.reviewer_1(item.id, reviewer="alice", text="Maakye")
    with pytest.raises(InvalidRequestError):
        store.reviewer_2(item.id, reviewer="alice", text="Maakye")
    reviewed = store.reviewer_2(item.id, reviewer="bob", text="Maakye")
    approved = store.approve(reviewed.id, approver="lead")
    assert approved.state == "approved"
    assert approved.approved_text == "Maakye"


def test_duplicate_audio_is_rejected(tmp_path: Path) -> None:
    """Normalized duplicate audio must not silently create repeated training rows."""
    store = CorpusStore(tmp_path)
    _item(store)
    with pytest.raises(ConflictError):
        _item(store)
