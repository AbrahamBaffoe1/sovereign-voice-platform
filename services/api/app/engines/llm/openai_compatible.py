"""Async client for a self-hosted OpenAI-compatible chat-completions server."""

from __future__ import annotations

import httpx

from app.config import Settings
from app.core.errors import ModelInferenceError
from app.engines.llm.base import LLMEngine


class OpenAICompatibleLocalLLM(LLMEngine):
    """Client for a local OpenAI-compatible endpoint such as vLLM or llama.cpp."""

    def __init__(self, settings: Settings) -> None:
        """Create one reusable asynchronous HTTP client for the configured local chat-completions
        endpoint; connection pooling matters for low-latency conversational turns."""
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url.rstrip("/") + "/",
            timeout=settings.llm_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    async def reply(self, text: str, *, language: str, system_prompt: str | None = None) -> str:
        """Build a bounded chat-completions request, call the local model server, validate the expected
        response shape, and translate transport/schema failures into one stable inference error."""
        prompt = system_prompt or (
            "You are the dialogue engine in a voice assistant. Respond naturally and concisely. "
            f"Use language code '{language}' unless the user clearly asks for another language. "
            "Do not include markdown unless the user asks for it."
        )
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        try:
            response = await self._client.post("chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM returned an empty response")
            return content.strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelInferenceError(f"local LLM inference failed: {exc}") from exc

    async def close(self) -> None:
        """Release the shared HTTP connection pool during application shutdown."""
        await self._client.aclose()
