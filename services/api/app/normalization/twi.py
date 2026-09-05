"""Orthography-preserving Twi normalization with reviewed-rule expansion intentionally gated."""

from __future__ import annotations

from app.normalization.orthography import OrthographyPreservingNormalizer


class TwiNormalizer(OrthographyPreservingNormalizer):
    """Preserve Twi/Akan spelling exactly while the pronunciation rule set is being reviewed.

    The shared Unicode normalizer already protects Twi letters, diacritics, names and code-switched
    text. Future number/date/currency expansion must be added here only after native-speaker golden
    tests exist; otherwise apparently helpful normalization can permanently poison TTS training data.
    """

    # TODO(language-tw): Add native-speaker-reviewed expansions for numbers, dates, time, GH₵,
    # abbreviations and code-switching. Every rule needs text->spoken-form golden tests first.
