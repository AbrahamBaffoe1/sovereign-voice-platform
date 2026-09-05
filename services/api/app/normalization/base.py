"""Abstract contract for language-aware text normalization before synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    @abstractmethod
    """Language-normalization interface used immediately before TTS. Implementations should preserve
    meaning and avoid pronunciation rules they cannot validate."""
    def normalize(self, text: str) -> str:
        """Transform raw generated text into the canonical written form expected by the target TTS
        tokenizer."""
        raise NotImplementedError
