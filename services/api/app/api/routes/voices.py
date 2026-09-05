"""HTTP endpoints for enumerating and enrolling local reference voices."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.errors import InvalidAudioError
from app.domain.models import VoiceProfilePublic
from app.services.audio import normalize_reference_audio_to_wav

router = APIRouter(prefix="/v1/voices", tags=["voices"])


@router.get("", response_model=list[VoiceProfilePublic])
async def list_voices(request: Request) -> list[VoiceProfilePublic]:
    """Return only the public portion of enrolled voice metadata; filesystem paths and engine-specific
    speaker IDs are intentionally not exposed."""
    return request.app.state.container.voices.list_public()


@router.post("/enroll", response_model=VoiceProfilePublic)
async def enroll_voice(
    request: Request,
    name: str = Form(...),
    language: str | None = Form(default=None),
    consent_attested: bool = Form(...),
    file: UploadFile = File(...),
) -> VoiceProfilePublic:
    """Validate language and upload bounds, decode the recording into a known mono WAV representation,
    require an explicit consent attestation, and persist the resulting local voice profile."""
    container = request.app.state.container
    if language is not None:
        container.languages.get(language)
    data = await file.read(container.settings.max_upload_bytes + 1)
    if not data or len(data) > container.settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="invalid or oversized voice reference")
    try:
        normalized_audio = normalize_reference_audio_to_wav(data)
        return container.voices.enroll_reference_audio(
            name=name,
            language=language,
            audio_bytes=normalized_audio,
            consent_attested=consent_attested,
        )
    except (ValueError, InvalidAudioError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
