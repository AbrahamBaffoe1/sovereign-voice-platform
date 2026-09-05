"""Composition root: creates the concrete engines, registries, router, and pipeline used by the API process."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.engines.asr.faster_whisper import FasterWhisperEngine
from app.engines.llm.openai_compatible import OpenAICompatibleLocalLLM
from app.normalization.registry import NormalizerRegistry
from app.orchestration.pipeline import VoicePipeline
from app.services.language_registry import LanguageRegistry
from app.services.tts_router import TTSRouter
from app.services.voice_registry import VoiceRegistry


@dataclass(slots=True)
class Container:
    """Explicit object graph for the running service. Route handlers reach shared engines through this
    container instead of constructing heavyweight models per request."""
    settings: Settings
    languages: LanguageRegistry
    voices: VoiceRegistry
    normalizers: NormalizerRegistry
    asr: FasterWhisperEngine
    llm: OpenAICompatibleLocalLLM | None
    tts: TTSRouter
    pipeline: VoicePipeline


def build_container(settings: Settings | None = None) -> Container:
    """Assemble one coherent runtime from configuration. This is the composition boundary where
    concrete adapters are chosen; orchestration code stays dependent on interfaces and registries."""
    settings = settings or get_settings()
    languages = LanguageRegistry(settings.language_config)
    voices = VoiceRegistry(settings.data_dir / "voices")
    normalizers = NormalizerRegistry()
    asr = FasterWhisperEngine(settings)
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
