# Architecture

## Runtime data path

```text
Client
  -> FastAPI ingress
  -> audio bounds/format validation
  -> FasterWhisperEngine
  -> normalized transcript
  -> optional LocalLLMEngine
  -> target LanguageSpec
  -> language normalizer
  -> VoiceRegistry lookup
  -> TTSRouter
       -> ChatterboxTTSEngine   (supported languages/reference voices)
       -> NemoFastPitchTTSEngine (our custom language checkpoints)
  -> WAV
  -> client
```

## Why two TTS backends

A multilingual pretrained model is valuable for immediate supported-language coverage and voice prompting. It is not evidence that an unsupported language is correctly modeled. Custom languages therefore use an independent checkpoint pair and a separately reviewed text frontend.

## Concurrency

ASR and each TTS engine use a single inference semaphore initially. GPU model calls are blocking and run via `asyncio.to_thread`, keeping the FastAPI event loop responsive. Model construction uses `AsyncLazy`, preventing two concurrent first requests from loading duplicate copies into VRAM.

Scale-out policy:

- one API worker per GPU-backed model process unless VRAM profiling proves safe otherwise;
- scale replicas horizontally behind a gateway;
- keep WebSocket sticky routing only for the lifetime of one socket;
- split ASR/TTS into separate GPU services when utilization profiles diverge;
- move model selection into a scheduler only after one-node metrics exist.

## Model versioning

Recommended production checkpoint layout:

```text
models/
  twi/
    v1.0.0/
      fastpitch.nemo
      hifigan.nemo
      model-card.json
      evaluation.json
    current -> v1.0.0
```

Do not overwrite a production checkpoint in place. Promote an immutable version and update the language config during deployment.

## Data privacy

The default architecture stores no conversation transcripts. Reference voices are stored locally. Add a database/event sink only if the product requires history; define retention before doing so.

## Latency roadmap

Current milestone is end-of-turn synthesis. Next low-latency milestone should be semantic end-of-turn/VAD plus incremental ASR. Do not sentence-chunk a non-streaming TTS model merely to claim streaming: discontinuities between independently synthesized chunks can hurt prosody and speaker consistency.
