"""Orthography-preserving Twi normalization that avoids inventing unreviewed pronunciation rules."""

from __future__ import annotations

import unicodedata

from app.normalization.generic import GenericNormalizer

_ALLOWED_NONLETTERS = set(" -’'.,;:!?₵$%()/:\"")


class TwiNormalizer(GenericNormalizer):
    """Conservative, orthography-preserving Twi normalization.

    We preserve every Unicode letter rather than maintaining an unreviewed alphabet
    in application code. This protects Twi graphemes, personal names, diacritics and
    code-switched words. The training pipeline separately audits the actual corpus
    inventory before a TTS tokenizer is frozen.
    """

    def normalize(self, text: str) -> str:
        """Preserve Twi letters, combining marks, numbers, and approved punctuation while replacing
        unsupported symbols with spaces. Final generic normalization cleans the resulting spacing
        without anglicizing the text."""
        value = unicodedata.normalize("NFC", text)
        value = value.replace("“", '"').replace("”", '"').replace("‘", "’")
        kept: list[str] = []
        for char in value:
            category = unicodedata.category(char)
            if category.startswith("L") or category.startswith("M") or category.startswith("N"):
                kept.append(char)
            elif char.isspace() or char in _ALLOWED_NONLETTERS:
                kept.append(char)
            else:
                kept.append(" ")
        return super().normalize("".join(kept))

        # TODO(intern-linguistics): Implement native-speaker-reviewed expansion for
        # cardinal/ordinal numbers, dates, time, GH₵ currency, abbreviations and
        # code-switching. Add golden pronunciation tests before enabling any rule.
