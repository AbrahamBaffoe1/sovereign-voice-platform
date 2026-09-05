"""Orchestration test that replaces heavyweight model engines with small deterministic fakes."""

from pathlib import Path

import pytest

from app.config import Settings
from app.domain.models import TranscriptionResult
from app.engines.asr.base import ASREngine
from app.engines.llm.base import LLMEngine
from app.engines.tts.base import TTSEngine
from app.normalization.registry import NormalizerRegistry
from app.orchestration.pipeline import VoicePipeline
from app.services.language_registry import LanguageRegistry
from app.services.voice_registry import VoiceRegistry


class FakeASR(ASREngine):
    """Internal FakeASR type used by the Orchestration test that replaces heavyweight model engines
    with small deterministic fakes."""
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> TranscriptionResult:
        """Implement the transcribe operation for this module while preserving its documented boundary
        behavior."""
        return TranscriptionResult(text="hello", language="en", language_probability=1.0)


class FakeLLM(LLMEngine):
    """Internal FakeLLM type used by the Orchestration test that replaces heavyweight model engines
    with small deterministic fakes."""
    async def reply(self, text: str, *, language: str, system_prompt: str | None = None) -> str:
        """Implement the reply operation for this module while preserving its documented boundary
        behavior."""
        return f"reply:{text}"


class FakeTTS(TTSEngine):
    """Internal FakeTTS type used by the Orchestration test that replaces heavyweight model engines
    with small deterministic fakes."""
    async def synthesize(self, text: str, *, language: str, voice=None, pace: float = 1.0):
        """Implement the synthesize operation for this module while preserving its documented boundary
        behavior."""
        assert text == "reply:hello"
        return b"RIFFfake", 24000


class FakeRouter:
    """Internal FakeRouter type used by the Orchestration test that replaces heavyweight model engines
    with small deterministic fakes."""
    def engine_for(self, language: str):
        """Implement the engine for operation for this module while preserving its documented boundary
        behavior."""
        return FakeTTS(), language


@pytest.mark.asyncio
async def test_pipeline_orchestrates(tmp_path: Path) -> None:
    """Regression test that verifies pipeline orchestrates. It protects this behavior from silent
    changes during refactors."""
    config = tmp_path / "languages.yaml"
    config.write_text(
        "languages:\n  en:\n    name: English\n    tts_engine: chatterbox\n    normalizer: generic\n",
        encoding="utf-8",
    )
    settings = Settings(language_config=config, data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    languages = LanguageRegistry(config)
    pipeline = VoicePipeline(
        settings=settings,
        asr=FakeASR(),
        llm=FakeLLM(),
        tts=FakeRouter(),
        languages=languages,
        normalizers=NormalizerRegistry(),
        voices=VoiceRegistry(tmp_path / "voices"),
    )
    result = await pipeline.handle_turn(
        b"audio",
        input_language="en",
        output_language="en",
        voice_id=None,
    )
    assert result.metadata.transcript == "hello"
    assert result.metadata.response_text == "reply:hello"
    assert result.metadata.audio_sample_rate == 24000
