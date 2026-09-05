# Sovereign Voice Platform

Self-hosted speech platform for ASR → optional local LLM → multilingual/custom-language TTS.

```text
Microphone / audio file
        ↓
Faster-Whisper (ASR)
        ↓
Local OpenAI-compatible LLM (optional)
        ↓
Language Router + Text Normalizer
        ↓
Chatterbox V3 OR custom NeMo FastPitch + HiFi-GAN
        ↓
WAV / PCM
```

## Architecture

The API is deliberately separated from model implementations. FastAPI owns HTTP/WebSocket transport; orchestration depends on ASR/LLM/TTS interfaces; model adapters own loading and inference; language configuration chooses normalizers and TTS backends. Heavy models load lazily and inference runs outside the event loop.

Repository layout:

```text
services/api/app/
  api/routes/           HTTP endpoints
  api/ws/               WebSocket voice-turn protocol
  engines/asr/          Faster-Whisper adapter
  engines/llm/          local OpenAI-compatible LLM adapter
  engines/tts/          Chatterbox + NeMo adapters
  normalization/        language-safe text normalization
  orchestration/        ASR -> LLM -> TTS pipeline
  services/             audio, voices, language registry, routing
training/
  common/               manifests + audio quality checks
  asr/                  Whisper fine-tuning/evaluation/export
  tts/                  NeMo FastPitch / HiFi-GAN launchers
  configs/languages/    reviewed language inventories
config/languages.yaml   runtime language routing
tests/                   regression/unit tests
docs/                    architecture and WebSocket protocol
examples/                HTTP/WebSocket clients
```

## Development

```bash
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[asr,dev]'
make run
```

Health check:

```bash
curl http://127.0.0.1:8080/healthz
```

Run validation:

```bash
make validate
make test
make lint
```

## TTS

Install Chatterbox for languages it natively supports:

```bash
pip install -e '.[tts-chatterbox]'
```

Example:

```bash
curl -X POST http://127.0.0.1:8080/v1/speech \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from my own voice server.","language":"en"}' \
  --output hello.wav
```

Custom languages use NeMo FastPitch + HiFi-GAN checkpoints. For Twi the runtime expects:

```text
models/twi/fastpitch.nemo
models/twi/hifigan.nemo
```

Do not point Twi/Ewe/Ga routes at English checkpoints merely to make them return audio. The tokenizer and pronunciation frontend must be reviewed for that language.

## Local LLM

The dialogue adapter expects an OpenAI-compatible `/v1/chat/completions` server, so vLLM, llama.cpp, or another compatible local server can be swapped without rewriting the voice pipeline.

```env
VOICE_LLM_BASE_URL=http://127.0.0.1:8001/v1
VOICE_LLM_MODEL=your-local-model
```

Disable the LLM to run ASR → TTS echo mode:

```env
VOICE_LLM_ENABLED=false
```

## WebSocket voice turns

Endpoint:

```text
ws://127.0.0.1:8080/v1/conversation
```

Send a JSON `start` frame, then mono PCM16 little-endian binary chunks, then `{"type":"commit"}`. The server returns transcript/response metadata followed by WAV bytes and `audio_end`. See `docs/WEBSOCKET_PROTOCOL.md` for the exact protocol.

## Dataset preparation

Expected CSV:

```csv
audio,text,speaker
001.wav,"Maakye, wo ho te sɛn?",spk01
002.wav,"Me ho yɛ, meda wo ase.",spk01
```

Compile and audit a corpus:

```bash
pip install -e '.[training]'
python -m training.prepare_dataset \
  --csv datasets/twi/metadata.csv \
  --audio-root datasets/twi/wavs \
  --output artifacts/twi \
  --language tw
```

The compiler checks audio readability, duration, clipping, RMS, DC offset, duplicate audio, transcript normalization, deterministic train/validation/test splits, and grapheme inventory.

## Whisper fine-tuning

```bash
pip install -e '.[training-asr]'
python -m training.asr.finetune_whisper \
  --train artifacts/twi/train.json \
  --validation artifacts/twi/validation.json \
  --output checkpoints/whisper-twi \
  --decoder-language <reviewed-token> \
  --fp16
```

Evaluate the final model on a frozen held-out set and convert the chosen checkpoint to CTranslate2 before deploying it through Faster-Whisper.

## Custom TTS training

Before FastPitch training: review the corpus inventory, freeze the grapheme/phoneme policy, define an explicit tokenizer, audit every transcript against it, then train the acoustic model and vocoder. The FastPitch launcher intentionally refuses configurations without an explicit tokenizer.

```bash
python -m training.tts.train_fastpitch --config path/to/reviewed-fastpitch.yaml --output checkpoints/twi-fastpitch
python -m training.tts.train_hifigan --config path/to/reviewed-hifigan.yaml --output checkpoints/twi-hifigan
```

## Before production exposure

Put the service behind TLS and an authenticated gateway; add tenant-aware rate limits; profile GPU concurrency; version checkpoints immutably; establish rollback; run native-speaker ASR/TTS evaluation; and load-test WebSocket/API behavior. `INTERN_TODOS.md` contains only delegated work that still requires human corpus, linguistic, evaluation, or deployment review.
