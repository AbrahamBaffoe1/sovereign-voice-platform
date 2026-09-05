"""Liveness and readiness endpoints used by humans, containers, and orchestration platforms."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return process liveness only. This endpoint deliberately avoids touching models or disk so an
    orchestrator can distinguish a live process from a fully ready deployment."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    """Expose lightweight readiness facts without forcing lazy GPU models to load. Operators can see
    configured languages, LLM availability, and whether ASR has already been materialized."""
    container = request.app.state.container
    return {
        "status": "ready",
        "asr_model_loaded": container.asr.model_loaded,
        "languages": [spec.code for spec in container.languages.all()],
        "llm_enabled": container.llm is not None,
    }
