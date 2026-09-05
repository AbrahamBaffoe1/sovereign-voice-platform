"""Read-only API for discovering the languages and TTS backends configured on this deployment."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/v1/languages", tags=["languages"])


@router.get("")
async def list_languages(request: Request) -> list[dict[str, object]]:
    """Return the deployment language catalog in API-safe form. A custom_model flag tells clients
    whether synthesis depends on a language-specific NeMo checkpoint rather than the shared general
    model."""
    return [
        {
            "code": spec.code,
            "name": spec.name,
            "tts_engine": spec.tts_engine,
            "custom_model": spec.tts_engine == "nemo",
        }
        for spec in request.app.state.container.languages.all()
    ]
