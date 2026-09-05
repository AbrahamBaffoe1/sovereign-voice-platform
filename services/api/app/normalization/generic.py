"""Conservative Unicode and whitespace normalization that is safe across languages."""

from __future__ import annotations

import re
import unicodedata

from app.normalization.base import TextNormalizer

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


class GenericNormalizer(TextNormalizer):
    """Conservative normalization shared by all languages.

    It intentionally does not verbalize numbers/dates because doing that without
    language-specific rules can corrupt pronunciation and semantic content.
    """

    def normalize(self, text: str) -> str:
        """Normalize Unicode composition, convert non-breaking spaces, collapse whitespace, and remove
        spaces before common punctuation without changing words or verbalizing numbers."""
        value = unicodedata.normalize("NFC", text)
        value = value.replace("\u00a0", " ")
        value = _WS.sub(" ", value).strip()
        value = _SPACE_BEFORE_PUNCT.sub(r"\1", value)
        return value
