"""Orthography-preserving Hausa normalization for Boko-script training and inference text."""

from __future__ import annotations

from app.normalization.orthography import OrthographyPreservingNormalizer


class HausaNormalizer(OrthographyPreservingNormalizer):
    """Preserve Hausa Boko orthography, including extended Latin letters, without anglicizing text."""

    # TODO(language-ha): Add native-speaker-reviewed handling for numbers, dates, currency,
    # abbreviations and Ajami/Boko transliteration policy. Do not auto-transliterate Ajami until that
    # separate data/linguistic decision has explicit coverage and tests.
