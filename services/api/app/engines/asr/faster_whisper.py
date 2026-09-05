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
    """Production adapter around one Faster-Whisper checkpoint.

    A process may host several instances of this class when different languages have their own
    fine-tuned CTranslate2 checkpoints. Each instance still loads lazily, so merely configuring four
    languages does not allocate four models into GPU memory at startup.
    """

    def __init__(self, settings: Settings, *, model_name: str | None = None) -> None:
        """Capture deployment settings and the checkpoint identifier owned by this engine instance."""
        self.settings = settings
        self.model_name = model_name or settings.asr_model
        self._model = AsyncLazy(self._load_model)
        self._inference_lock = asyncio.Semaphore(1)

    @property
    def model_loaded(self) -> bool:
        """Expose lazy-load state for readiness diagnostics without touching model weights."""
        return self._model.loaded

    async def _load_model(self) -> Any:
        """Import Faster-Whisper only on first use and deserialize this instance's checkpoint off-loop."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EngineUnavailableError(
                "faster-whisper is not installed; install the 'asr' extra"
            ) from exc

        def load() -> Any:
            """Construct the synchronous model in a worker thread so startup does not block FastAPI."""
            return WhisperModel(
                self.model_name,
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
        """Decode one request and convert model-specific segments into the platform result contract."""
        model = await self._model.get()

        def infer(path: str) -> TranscriptionResult:
            """Run synchronous CTranslate2 inference and consume the lazy segment iterator completely."""
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
                            probability=(
                                float(word.probability) if word.probability is not None else None
                            ),
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

        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = Path(tmp.name)
            async with self._inference_lock:
                return await asyncio.to_thread(infer, str(path))
        except Exception as exc:
            if isinstance(exc, EngineUnavailableError):
                raise
            raise ModelInferenceError(f"ASR inference failed for {self.model_name!r}: {exc}") from exc
        finally:
            if path:
                path.unlink(missing_ok=True)
