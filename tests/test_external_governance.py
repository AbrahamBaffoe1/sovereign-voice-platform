"""Tests for truthful first-party versus licensed-external corpus admission rules."""

from training.prepare_dataset import _governance_ok, _review_ok


def test_first_party_or_licensed_external_governance_is_accepted() -> None:
    """External public data should not need a fake first-party consent flag when its source is approved."""
    assert _governance_ok(consent_attested=True, governance_approved=False)
    assert _governance_ok(consent_attested=False, governance_approved=True)
    assert not _governance_ok(consent_attested=False, governance_approved=False)


def test_local_review_or_upstream_validation_is_accepted() -> None:
    """The audit trail distinguishes our human review from an approved upstream validation process."""
    assert _review_ok(transcript_reviewed=True, upstream_validated=False)
    assert _review_ok(transcript_reviewed=False, upstream_validated=True)
    assert not _review_ok(transcript_reviewed=False, upstream_validated=False)
