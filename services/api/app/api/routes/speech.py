"""HTTP speech endpoints for bounded audio transcription and text-to-speech synthesis."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from app.core.errors import ResourceNotFoundError
from app.domain.models import SpeechRequest, TranscriptionResult

router = APIRouter(prefix="/v1", tags=["speech"])


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read at most one byte beyond the configured upload limit so oversized requests can be rejected
    without buffering an unbounded body in memory. Empty uploads are rejected before they reach a
    decoder."""
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail="audio upload exceeds configured limit")
    if not data:
        raise HTTPException(status_code=400, detail="audio upload is empty")
    return data


@router.post("/transcriptions", response_model=TranscriptionResult)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    hotwords: str | None = Form(default=None),
    word_timestamps: bool = Form(default=False),
) -> TranscriptionResult:
    """Validate the uploaded audio size, then delegate recognition to the ASR engine with optional
    language hints, hotwords, and word timestamps. Model-specific details remain behind the engine
    interface."""
    container = request.app.state.container
    audio = await _read_limited(file, container.settings.max_upload_bytes)
    return await container.asr.transcribe(
        audio,
        language=language,
        hotwords=hotwords,
        word_timestamps=word_timestamps,
    )


@router.post("/speech")
async def synthesize(body: SpeechRequest, request: Request) -> Response:
    """Resolve the language, run its configured normalizer, optionally resolve a stored voice, route to
    the correct TTS engine, and return real WAV bytes with the sample rate in a response header."""
    container = request.app.state.container
    language_spec = container.languages.get(body.language)
    text = container.normalizers.get(language_spec.normalizer).normalize(body.text)
    voice = container.voices.get(body.voice_id) if body.voice_id else None
    if body.voice_id and voice is None:
        raise ResourceNotFoundError(f"voice not found: {body.voice_id}")
    engine, engine_language = container.tts.engine_for(body.language)
    wav, sample_rate = await engine.synthesize(
        text,
        language=engine_language,
        voice=voice,
        pace=body.pace,
    )
    return Response(
        wav,
        media_type="audio/wav",
        headers={"X-Audio-Sample-Rate": str(sample_rate)},
    )
