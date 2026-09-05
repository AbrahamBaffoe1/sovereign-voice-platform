# Architecture

## Runtime request path

```text
client audio
   |
   v
FastAPI / WebSocket
   |
   v
ASRRouter
   |-- no language / shared languages --> shared Faster-Whisper
   |-- tw  --> models/asr/tw
   |-- gaa --> models/asr/gaa
   |-- ee  --> models/asr/ee
   `-- ha  --> models/asr/ha
   |
   v
optional local OpenAI-compatible LLM
   |
   v
LanguageRegistry -> language normalizer
   |
   v
TTSRouter
   |-- Chatterbox for configured native languages
   `-- per-language NeMo FastPitch + HiFi-GAN for custom languages
   |
   v
WAV / PCM delivery
```

Explicit Twi, Ga, Ewe and Hausa requests are strict custom-model routes. Missing checkpoints are deployment errors rather than reasons to silently use generic models.

## Training-data path

```text
consented recordings + reviewed transcripts
       -> training.prepare_dataset
       -> train/validation/test + audit + inventory + quality report
       -> dataset_version.json
       -> ASR/TTS experiments
       -> evaluation
       -> model_card.json
```

Strict target-language corpora use speaker IDs as deterministic split keys, preventing the same voice from leaking into training and held-out evaluation.

Runtime routing is in `config/languages.yaml`. Training policy is in `training/configs/languages/*.yaml`. TTS preflight blocks production training until a native-speaker-reviewed grapheme/tokenizer policy exists.
