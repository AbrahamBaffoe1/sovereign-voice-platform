"""Read-only API for inspecting configured languages, aliases, engines and deployment readiness."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(prefix="/v1/languages", tags=["languages"])


@router.get("")
async def list_languages(request: Request) -> list[dict[str, object]]:
    """Return model routing and checkpoint readiness without loading any heavyweight model."""
    container = request.app.state.container
    output: list[dict[str, object]] = []
    for spec in container.languages.all():
        asr = spec.asr or {"mode": "shared"}
        asr_mode = str(asr.get("mode", "shared"))
        asr_model = str(asr.get("model", container.settings.asr_model))
        asr_ready = asr_mode == "shared" or str(asr.get("model_kind", "local")) == "remote"
        if asr_mode == "custom" and str(asr.get("model_kind", "local")) == "local":
            asr_ready = Path(asr_model).exists()
        tts_ready = True
        if spec.tts_engine == "nemo":
            nemo = spec.nemo or {}
            fastpitch_raw = str(nemo.get("fastpitch_checkpoint", "")).strip()
            hifigan_raw = str(nemo.get("hifigan_checkpoint", "")).strip()
            tts_ready = bool(fastpitch_raw and hifigan_raw)
            if tts_ready:
                tts_ready = Path(fastpitch_raw).exists() and Path(hifigan_raw).exists()
        output.append(
            {
                "code": spec.code,
                "name": spec.name,
                "iso639_3": spec.iso639_3,
                "aliases": list(spec.aliases),
                "normalizer": spec.normalizer,
                "asr": {"mode": asr_mode, "model": asr_model, "ready": asr_ready},
                "tts": {"engine": spec.tts_engine, "ready": tts_ready},
                "training_profile": spec.training_profile,
            }
        )
    return output
