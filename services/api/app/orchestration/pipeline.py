"""Application service that coordinates ASR, optional dialogue generation, normalization, voice selection, and TTS."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.config import Settings
from app.core.errors import ResourceNotFoundError
from app.domain.models import ConversationTurnResult
from app.engines.asr.base import ASREngine
from app.engines.llm.base import LLMEngine
from app.normalization.registry import NormalizerRegistry
from app.services.language_registry import LanguageRegistry
from app.services.tts_router import TTSRouter
from app.services.voice_registry import VoiceRegistry


@dataclass(slots=True)
class VoiceTurn:
    """Internal return type that keeps JSON-friendly turn metadata beside the binary WAV payload."""
    metadata: ConversationTurnResult
    wav_bytes: bytes


class VoicePipeline:
    """Coordinates one conversational turn while keeping each model backend replaceable. It is
    intentionally free of HTTP/WebSocket concerns."""
    def __init__(
        self,
        *,
        settings: Settings,
        asr: ASREngine,
        llm: LLMEngine | None,
        tts: TTSRouter,
        languages: LanguageRegistry,
        normalizers: NormalizerRegistry,
        voices: VoiceRegistry,
    ) -> None:
        """Capture shared runtime collaborators once; no heavyweight model is instantiated here because
        each engine owns its own lifecycle."""
        self.settings = settings
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.languages = languages
        self.normalizers = normalizers
        self.voices = voices

    async def handle_turn(
        self,
        audio_bytes: bytes,
        *,
        input_language: str | None,
        output_language: str | None,
        voice_id: str | None,
        hotwords: str | None = None,
        system_prompt: str | None = None,
    ) -> VoiceTurn:
        """Execute the ordered ASR -> optional LLM -> language normalization -> voice resolution -> TTS
        workflow, measure stage latency, and return both response metadata and synthesized audio."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        transcription = await self.asr.transcribe(
            audio_bytes,
            language=input_language,
            hotwords=hotwords,
            word_timestamps=False,
        )
        timings["asr"] = (time.perf_counter() - started) * 1000

        detected_language = input_language or transcription.language
        target_language = output_language or detected_language
        target_spec = self.languages.get(target_language)

        llm_started = time.perf_counter()
        if self.llm is None:
            response_text = transcription.text
        else:
            response_text = await self.llm.reply(
                transcription.text,
                language=target_language,
                system_prompt=system_prompt,
            )
        timings["llm"] = (time.perf_counter() - llm_started) * 1000

        normalized = self.normalizers.get(target_spec.normalizer).normalize(response_text)
        voice = self.voices.get(voice_id) if voice_id else None
        if voice_id and voice is None:
            raise ResourceNotFoundError(f"voice not found: {voice_id}")
        tts_engine, engine_language = self.tts.engine_for(target_language)

        tts_started = time.perf_counter()
        wav_bytes, sample_rate = await tts_engine.synthesize(
            normalized,
            language=engine_language,
            voice=voice,
        )
        timings["tts"] = (time.perf_counter() - tts_started) * 1000
        timings["total"] = (time.perf_counter() - started) * 1000

        return VoiceTurn(
            metadata=ConversationTurnResult(
                transcript=transcription.text,
                input_language=detected_language,
                response_text=response_text,
                output_language=target_language,
                audio_sample_rate=sample_rate,
                timings_ms={key: round(value, 2) for key, value in timings.items()},
            ),
            wav_bytes=wav_bytes,
        )
