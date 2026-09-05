"""Abstract dialogue-engine contract used by the voice pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMEngine(ABC):
    """Interface boundary for the optional dialogue stage between ASR and TTS."""

    @abstractmethod
    async def reply(self, text: str, *, language: str, system_prompt: str | None = None) -> str:
        """Produce response text without exposing a particular local model server to orchestration."""
        raise NotImplementedError
