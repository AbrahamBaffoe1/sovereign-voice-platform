# Sovereign Voice Platform

Self-hosted `ASR -> optional LLM -> TTS` for English plus custom Twi, Ga, Ewe and Hausa models.

## Runtime

```text
microphone/audio
    -> ASRRouter
       -> shared Faster-Whisper for generic discovery
       -> custom CTranslate2 checkpoint for tw / gaa / ee / ha
    -> local OpenAI-compatible LLM (optional)
    -> language normalizer
    -> TTSRouter
       -> Chatterbox for configured native languages
       -> NeMo FastPitch + HiFi-GAN for custom languages
    -> WAV
```

Custom-language routes are strict. If a Twi/Ga/Ewe/Hausa checkpoint is missing, the API returns a deployment error instead of quietly using the wrong model.

## Setup

```bash
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[asr,dev]'
make run
```

Optional model dependencies:

```bash
pip install -e '.[tts-chatterbox]'
pip install -e '.[tts-nemo]'
pip install -e '.[training,training-asr]'
```

## Target language codes

| Language | Canonical code | Accepted aliases |
|---|---|---|
| Twi | `tw` | `twi`, `akan-twi` |
| Ga | `gaa` | `ga` |
| Ewe | `ee` | `ewe` |
| Hausa | `ha` | `hausa` |

Runtime routing: `config/languages.yaml`  
Training policy: `training/configs/languages/*.yaml`

Check deployment readiness without loading models:

```bash
curl http://127.0.0.1:8080/v1/languages
```

## Corpus workflow

Strict workspaces live under `datasets/tw`, `datasets/gaa`, `datasets/ee`, and `datasets/ha`. CSV schema:

```text
audio,text,speaker,dialect,source_id,consent_attested,transcript_reviewed
```

Compile a corpus:

```bash
python -m training.prepare_dataset \
  --profile training/configs/languages/tw.yaml \
  --csv datasets/tw/metadata.csv \
  --audio-root datasets/tw/wavs \
  --output artifacts/tw
```

The compiler enforces consent/review metadata, speaker-disjoint splits and emits `audit.jsonl`, `inventory.json`, `quality_report.json`, and `dataset_version.json`.

## ASR training

```bash
python -m training.asr.finetune_whisper \
  --profile training/configs/languages/tw.yaml \
  --train artifacts/tw/train.json \
  --validation artifacts/tw/validation.json \
  --output checkpoints/whisper-tw \
  --fp16
```

Profiles currently use token-free low-resource experiments rather than inventing unsupported Whisper language tokens.

## TTS training gate

```bash
python -m training.tts.preflight \
  --profile training/configs/languages/tw.yaml \
  --artifacts artifacts/tw
```

It intentionally fails until native-speaker/linguist review approves the grapheme/tokenizer policy.

## Validation

```bash
python -m compileall -q services training tests
pytest -q
```

Current execution plan: `docs/EXECUTION_BACKLOG.md`.
