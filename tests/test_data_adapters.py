"""Provider adapter tests for speaker-leakage safeguards."""

from training.data.adapters import adapt_common_voice, adapt_ga_parallel, adapt_waxal


def test_ga_rows_without_speaker_are_training_only() -> None:
    """The adapter must not fabricate a speaker identity merely to satisfy split tooling."""
    row = adapt_ga_parallel({"audio": "x.wav", "text": "hello", "id": "1"})
    assert row.speaker is None
    assert row.training_only is True


def test_speaker_aware_sources_keep_real_ids() -> None:
    """Speaker identifiers must survive adaptation for disjoint splits."""
    waxal = adapt_waxal({"audio": "a.wav", "text": "x", "speaker_id": "spk-a"})
    cv = adapt_common_voice({"audio": "b.wav", "sentence": "x", "client_id": "spk-b"})
    assert waxal.speaker == "spk-a"
    assert cv.speaker == "spk-b"
    assert not cv.training_only
