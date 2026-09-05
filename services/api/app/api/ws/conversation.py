"""Stateful WebSocket protocol that turns streamed PCM microphone bytes into one conversational voice turn."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.core.errors import VoicePlatformError
from app.core.security import authorize_websocket
from app.services.audio import pcm16_mono_to_wav_bytes, pcm_duration_seconds

router = APIRouter(tags=["conversation"])
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TurnBuffer:
    """Mutable state for exactly one in-progress WebSocket speech turn. It stores protocol metadata
    separately from raw PCM chunks and tracks bytes incrementally so duration limits can be enforced
    before concatenation."""
    sample_rate: int = 16000
    input_language: str | None = None
    output_language: str | None = None
    voice_id: str | None = None
    hotwords: str | None = None
    system_prompt: str | None = None
    chunks: list[bytes] = field(default_factory=list)
    byte_count: int = 0

    def append(self, data: bytes, max_seconds: int) -> None:
        """Accept one raw PCM chunk only if adding it keeps the turn inside the configured duration
        ceiling. The byte counter makes this O(1) per frame instead of repeatedly joining all
        previous chunks."""
        duration = pcm_duration_seconds(self.byte_count + len(data), self.sample_rate)
        if duration > max_seconds:
            raise ValueError(f"audio exceeds {max_seconds}s WebSocket turn limit")
        self.chunks.append(data)
        self.byte_count += len(data)

    def wav(self) -> bytes:
        """Commit the buffered PCM turn into a standard WAV container. Empty turns are rejected because
        ASR backends should never receive an ambiguous zero-length recording."""
        if not self.chunks:
            raise ValueError("cannot commit an empty audio turn")
        return pcm16_mono_to_wav_bytes(b"".join(self.chunks), self.sample_rate)


@router.websocket("/v1/conversation")
async def conversation_socket(websocket: WebSocket) -> None:
    """Run the WebSocket state machine. JSON frames control start/cancel/commit, binary frames carry
    PCM16 audio, and every committed turn is reset in a finally block so failures cannot leak stale
    audio into the next turn."""
    container = websocket.app.state.container
    if not await authorize_websocket(websocket, container.settings):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    state: TurnBuffer | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                if state is None:
                    await websocket.send_json({"type": "error", "error": "send start before audio"})
                    continue
                try:
                    state.append(message["bytes"], container.settings.max_ws_audio_seconds)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                    state = None
                continue

            text = message.get("text")
            if text is None:
                continue
            try:
                import orjson

                event = orjson.loads(text)
            except Exception:
                await websocket.send_json({"type": "error", "error": "invalid JSON control frame"})
                continue

            event_type = event.get("type")
            if event_type == "start":
                sample_rate = int(event.get("sample_rate", container.settings.default_sample_rate))
                if not 8000 <= sample_rate <= 48000:
                    await websocket.send_json({"type": "error", "error": "sample_rate must be 8000-48000"})
                    continue
                state = TurnBuffer(
                    sample_rate=sample_rate,
                    input_language=event.get("input_language"),
                    output_language=event.get("output_language"),
                    voice_id=event.get("voice_id"),
                    hotwords=event.get("hotwords"),
                    system_prompt=event.get("system_prompt"),
                )
                await websocket.send_json({"type": "started"})
            elif event_type == "cancel":
                state = None
                await websocket.send_json({"type": "cancelled"})
            elif event_type == "commit":
                if state is None:
                    await websocket.send_json({"type": "error", "error": "no active turn"})
                    continue
                try:
                    wav = state.wav()
                    await websocket.send_json({"type": "processing"})
                    result = await container.pipeline.handle_turn(
                        wav,
                        input_language=state.input_language,
                        output_language=state.output_language,
                        voice_id=state.voice_id,
                        hotwords=state.hotwords,
                        system_prompt=state.system_prompt,
                    )
                    await websocket.send_json({"type": "result", **result.metadata.model_dump()})
                    await websocket.send_bytes(result.wav_bytes)
                    await websocket.send_json({"type": "audio_end"})
                except VoicePlatformError as exc:
                    await websocket.send_json({"type": "error", "error": str(exc)})
                except Exception:
                    logger.exception("unhandled WebSocket voice-turn failure")
                    await websocket.send_json({"type": "error", "error": "internal processing error"})
                finally:
                    state = None
            else:
                await websocket.send_json({"type": "error", "error": f"unknown event type: {event_type}"})
    except WebSocketDisconnect:
        return
