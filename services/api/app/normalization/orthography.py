"""Shared Unicode-safe normalization for languages whose spoken-text rules are still under review."""

from __future__ import annotations

import unicodedata

from app.normalization.generic import GenericNormalizer

_ALLOWED_NONLETTERS = set(" -–—’'.,;:!?₵$%()/:\"+@#&")


class OrthographyPreservingNormalizer(GenericNormalizer):
    """Normalize Unicode and spacing while preserving the language's actual written symbols.

    This class deliberately avoids maintaining an alphabet allow-list. During early language work,
    an incomplete allow-list is dangerous: one missing letter silently corrupts training text and
    teaches the TTS model a bad mapping. The training preflight separately audits observed graphemes
    against a native-speaker-reviewed inventory before a production tokenizer may be frozen.
    """

    def normalize(self, text: str) -> str:
        """Keep letters, combining marks, numbers and conservative punctuation; replace unrelated
        symbols with spaces, then delegate whitespace/punctuation cleanup to GenericNormalizer."""
        value = unicodedata.normalize("NFC", text)
        value = value.replace("“", '"').replace("”", '"').replace("‘", "’")
        kept: list[str] = []
        for char in value:
            category = unicodedata.category(char)
            if category.startswith(("L", "M", "N")):
                kept.append(char)
            elif char.isspace() or char in _ALLOWED_NONLETTERS:
                kept.append(char)
            else:
                kept.append(" ")
        return super().normalize("".join(kept))
