"""Composition root: creates concrete engines, registries, routers, and orchestration services."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.engines.llm.openai_compatible import OpenAICompatibleLocalLLM
from app.normalization.registry import NormalizerRegistry
from app.orchestration.pipeline import VoicePipeline
from app.services.asr_router import ASRRouter
from app.services.language_registry import LanguageRegistry
from app.services.tts_router import TTSRouter
from app.services.voice_registry import VoiceRegistry


@dataclass(slots=True)
class Container:
    """Explicit object graph shared by route handlers for the lifetime of the API process."""

    settings: Settings
    languages: LanguageRegistry
    voices: VoiceRegistry
    normalizers: NormalizerRegistry
    asr: ASRRouter
    llm: OpenAICompatibleLocalLLM | None
    tts: TTSRouter
    pipeline: VoicePipeline


def build_container(settings: Settings | None = None) -> Container:
    """Assemble runtime adapters once while preserving lazy loading inside heavyweight model engines."""
    settings = settings or get_settings()
    languages = LanguageRegistry(settings.language_config)
    voices = VoiceRegistry(settings.data_dir / "voices")
    normalizers = NormalizerRegistry()
    asr = ASRRouter(settings, languages)
    llm = OpenAICompatibleLocalLLM(settings) if settings.llm_enabled else None
    tts = TTSRouter(settings, languages)
    pipeline = VoicePipeline(
        settings=settings,
        asr=asr,
        llm=llm,
        tts=tts,
        languages=languages,
        normalizers=normalizers,
        voices=voices,
    )
    return Container(settings, languages, voices, normalizers, asr, llm, tts, pipeline)
