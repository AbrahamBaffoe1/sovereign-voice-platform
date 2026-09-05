"""Prometheus metrics for speech turns and optional local GPU memory telemetry."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

TURN_TOTAL = Counter(
    "voice_turns_total",
    "Completed or failed conversational turns",
    ["status", "input_language", "output_language"],
)
TURN_STAGE_SECONDS = Histogram(
    "voice_turn_stage_seconds",
    "Latency of ASR, LLM, TTS and whole-turn stages",
    ["stage", "input_language", "output_language"],
)
TTS_AUDIO_BYTES = Histogram(
    "voice_tts_audio_bytes",
    "Size of synthesized WAV output per completed turn",
    buckets=(4096, 16384, 65536, 262144, 1048576, 4194304, float("inf")),
)
GPU_MEMORY_ALLOCATED = Gauge(
    "voice_gpu_memory_allocated_bytes",
    "Current torch CUDA allocated bytes",
    ["device"],
)
GPU_MEMORY_RESERVED = Gauge(
    "voice_gpu_memory_reserved_bytes",
    "Current torch CUDA reserved bytes",
    ["device"],
)


def observe_turn(
    *,
    input_language: str,
    output_language: str,
    timings_ms: dict[str, float],
    audio_bytes: int,
) -> None:
    """Record successful stage latency without allowing metric code to change model results."""
    for stage, milliseconds in timings_ms.items():
        TURN_STAGE_SECONDS.labels(stage, input_language, output_language).observe(
            milliseconds / 1000.0
        )
    TTS_AUDIO_BYTES.observe(audio_bytes)
    TURN_TOTAL.labels("ok", input_language, output_language).inc()


def observe_turn_failure(*, input_language: str | None, output_language: str | None) -> None:
    """Increment a bounded-cardinality failure counter using requested language labels when known."""
    TURN_TOTAL.labels("error", input_language or "auto", output_language or "auto").inc()


def refresh_gpu_metrics() -> None:
    """Refresh CUDA memory gauges when torch/CUDA exists; CPU deployments remain dependency-free."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        label = f"cuda:{index}"
        GPU_MEMORY_ALLOCATED.labels(label).set(torch.cuda.memory_allocated(index))
        GPU_MEMORY_RESERVED.labels(label).set(torch.cuda.memory_reserved(index))
