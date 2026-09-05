"""Orthography-preserving Ga normalization; spoken-form transformations remain review-gated."""

from __future__ import annotations

from app.normalization.orthography import OrthographyPreservingNormalizer


class GaNormalizer(OrthographyPreservingNormalizer):
    """Protect Ga graphemes and diacritics while the production pronunciation policy is reviewed."""

    # TODO(language-gaa): Add native-speaker-reviewed number/date/currency/abbreviation rules plus
    # regression fixtures before enabling any transformation beyond Unicode/spacing cleanup.
