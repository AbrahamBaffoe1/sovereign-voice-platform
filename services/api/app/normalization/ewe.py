"""Orthography-preserving Ewe normalization; spoken-form transformations remain review-gated."""

from __future__ import annotations

from app.normalization.orthography import OrthographyPreservingNormalizer


class EweNormalizer(OrthographyPreservingNormalizer):
    """Protect Ewe graphemes and diacritics without importing English pronunciation assumptions."""

    # TODO(language-ee): Implement reviewed Ewe number/date/currency/abbreviation expansion and add
    # golden tests created with native speakers before enabling the transformations at runtime.
