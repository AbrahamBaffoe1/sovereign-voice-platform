"""Small authentication helpers shared by HTTP and WebSocket entry points."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, WebSocket, status

from app.config import Settings


def require_api_key(settings: Settings):
    """Create a FastAPI dependency bound to this deployment configuration. Authentication becomes a
    no-op when no API key is configured, which keeps local development friction low."""
    async def dependency(x_voice_api_key: str | None = Header(default=None)) -> None:
        """Validate the X-Voice-API-Key header using constant-time comparison when authentication is
        enabled."""
        if not settings.api_key:
            return
        if not x_voice_api_key or not hmac.compare_digest(x_voice_api_key, settings.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")

    return dependency


async def authorize_websocket(websocket: WebSocket, settings: Settings) -> bool:
    """Apply the same API-key policy to WebSocket handshakes, accepting either a header or query value
    because browser WebSocket clients cannot set arbitrary headers reliably."""
    if not settings.api_key:
        return True
    supplied = websocket.headers.get("x-voice-api-key") or websocket.query_params.get("api_key")
    return bool(supplied and hmac.compare_digest(supplied, settings.api_key))
