"""Self-hosted Faster-Whisper adapter with lazy model loading and serialized inference."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.errors import EngineUnavailableError, ModelInferenceError
from app.core.lifecycle import AsyncLazy
from app.domain.models import TranscriptionResult, WordTimestamp
from app.engines.asr.base import ASREngine


class FasterWhisperEngine(ASREngine):
    """Production adapter around Faster-Whisper. Model construction is lazy and inference is serialized
    by default to avoid uncontrolled concurrent GPU memory pressure."""
    def __init__(self, settings: Settings) -> None:
        """Capture deployment settings, wrap model construction in AsyncLazy, and create the inference
        semaphore that protects a single model instance."""
        self.settings = settings
        self._model = AsyncLazy(self._load_model)
        self._inference_lock = asyncio.Semaphore(1)

    @property
    def model_loaded(self) -> bool:
        """Expose lazy-load state for readiness diagnostics without allocating model memory."""
        return self._model.loaded

    async def _load_model(self) -> Any:
        """Import the optional Faster-Whisper dependency only when ASR is first used, choose the
        configured device/compute mode, and load the model off the event loop."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EngineUnavailableError(
                "faster-whisper is not installed; install the 'asr' extra"
            ) from exc

        def load() -> Any:
            """Construct the synchronous WhisperModel object inside a worker thread so model
            download/deserialization does not block FastAPI event processing."""
            return WhisperModel(
                self.settings.asr_model,
                device=self.settings.asr_device,
                compute_type=self.settings.asr_compute_type,
            )

        return await asyncio.to_thread(load)

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        hotwords: str | None = None,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Materialize the model, stage request bytes in a temporary file for the decoder, run blocking
        recognition outside the event loop, map segments into engine-neutral models, and always
        delete the temporary file."""
        model = await self._model.get()

        def infer(path: str) -> TranscriptionResult:
            """Execute the synchronous Faster-Whisper call and collapse lazy segment generators into
            text plus optional per-word timestamps while model metadata is still available."""
            segments, info = model.transcribe(
                path,
                language=language,
                beam_size=self.settings.asr_beam_size,
                vad_filter=self.settings.asr_vad,
                word_timestamps=word_timestamps,
                hotwords=hotwords,
                condition_on_previous_text=False,
            )
            text_parts: list[str] = []
            words: list[WordTimestamp] = []
            for segment in segments:
                text_parts.append(segment.text.strip())
                if word_timestamps and segment.words:
                    words.extend(
                        WordTimestamp(
                            word=word.word,
                            start=float(word.start),
                            end=float(word.end),
                            probability=(float(word.probability) if word.probability is not None else None),
                        )
                        for word in segment.words
                    )
            return TranscriptionResult(
                text=" ".join(part for part in text_parts if part).strip(),
                language=info.language,
                language_probability=float(info.language_probability),
                duration_seconds=float(info.duration),
                words=words,
            )

        suffix = ".audio"
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                path = Path(tmp.name)
            async with self._inference_lock:
                return await asyncio.to_thread(infer, str(path))
        except Exception as exc:
            if isinstance(exc, EngineUnavailableError):
                raise
            raise ModelInferenceError(f"ASR inference failed: {exc}") from exc
        finally:
            if path:
                path.unlink(missing_ok=True)
