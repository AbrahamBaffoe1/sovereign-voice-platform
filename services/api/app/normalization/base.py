"""Abstract contract for language-aware text normalization before synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    """Language-normalization boundary used immediately before text reaches a TTS tokenizer."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Transform generated text without introducing pronunciation rules the implementation cannot validate."""
        raise NotImplementedError
