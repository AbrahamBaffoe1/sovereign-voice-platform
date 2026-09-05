"""Regression tests for consent enforcement and filesystem voice enrollment."""

from pathlib import Path

import pytest

from app.services.voice_registry import VoiceRegistry


def test_voice_enrollment_requires_consent(tmp_path: Path) -> None:
    """Regression test that verifies voice enrollment requires consent. It protects this behavior from
    silent changes during refactors."""
    registry = VoiceRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.enroll_reference_audio(
            name="Test",
            language="en",
            audio_bytes=b"abc",
            consent_attested=False,
        )


def test_voice_enrollment_roundtrip(tmp_path: Path) -> None:
    """Regression test that verifies voice enrollment roundtrip. It protects this behavior from silent
    changes during refactors."""
    registry = VoiceRegistry(tmp_path)
    public = registry.enroll_reference_audio(
        name="Test",
        language="en",
        audio_bytes=b"abc",
        consent_attested=True,
    )
    profile = registry.get(public.id)
    assert profile is not None
    assert profile.name == "Test"
    assert profile.reference_audio_path is not None
