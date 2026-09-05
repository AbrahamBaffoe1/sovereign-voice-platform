# Sovereign Voice Platform

Self-hosted speech stack for **Twi (`tw`)**, **Ga (`gaa`)**, **Ewe (`ee`)**, **Hausa (`ha`)**, and configured general-purpose languages.

The repository contains runtime inference, governed corpus intake, ASR/TTS training utilities, speech benchmarks, model promotion/rollback, and a TypeScript client SDK. It does not contain private training audio or model weights.

## Runtime

```text
microphone / uploaded audio
        |
        v
Faster-Whisper / promoted language ASR
        |
        v
optional local OpenAI-compatible LLM
        |
        v
language normalizer
        |
        v
TTS router
  |- Chatterbox for configured supported languages
  |- NeMo FastPitch + HiFi-GAN baseline
  `- VoxCPM2 adapted checkpoint candidate
        |
        v
WAV / PCM
```

For `tw`, `gaa`, `ee`, and `ha`, custom ASR checkpoints are language-specific and must be deployed explicitly. A missing required checkpoint is an error; the runtime does not silently substitute a generic model for an explicit target-language request.

## Install

```bash
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[asr,dev]'
make run
```

Install only the model stacks you need:

```bash
pip install -e '.[tts-chatterbox]'
pip install -e '.[tts-nemo]'
pip install -e '.[tts-voxcpm]'
pip install -e '.[training,training-asr]'
```

## Corpus intake

Use `POST /v1/corpus/items` for an already-cut speech clip and `POST /v1/corpus/recordings` for a longer team recording or voice-message container. Long recordings are decoded, normalized, segmented, and placed into the same review workflow. Common phone formats can fall back to bounded `ffmpeg` decoding.

Transcript states are:

```text
machine_draft
    -> reviewer_1_complete
    -> reviewer_2_complete
    -> approved
```

Rejected items remain in the audit trail. Two independent reviewer passes are required before approved export.

Approved data can be exported into the strict training schema:

```text
audio,text,speaker,dialect,source_id,consent_attested,transcript_reviewed
```

The compiler creates speaker-disjoint splits, an audit manifest, quality report, grapheme inventory, rejected-row report, and deterministic `dataset_version.json` fingerprint.

```bash
python -m training.prepare_dataset \
  --profile training/configs/languages/tw.yaml \
  --csv datasets/tw/metadata.csv \
  --audio-root datasets/tw/wavs \
  --output artifacts/tw
```

## First-party team recordings

Fill `datasets/team_recordings_inventory.csv` with original file paths and provenance, then run:

```bash
python -m training.ingest.import_team_recordings \
  --inventory datasets/team_recordings_inventory.csv \
  --input-root /path/to/original-recordings \
  --corpus-root data/corpus
```

Single-speaker recordings are normalized and segmented automatically. Rows marked `multi_speaker=true` are quarantined for diarization rather than mislabeled as one speaker.

Machine drafts are optional and never approve themselves:

```bash
python -m training.ingest.machine_drafts \
  --corpus-root data/corpus \
  --model large-v3
```

## External data policy

`training/configs/source_catalog.yaml` is the source of truth for whether an external corpus is allowed for production training, evaluation only, or research only.

Plan sources without downloading anything:

```bash
python -m training.data.plan_sources --language tw --usage production
```

A Hugging Face snapshot used for a governed experiment must be pinned to an explicit revision when the catalog requires it:

```bash
python -m training.data.snapshot_hf \
  --catalog training/configs/source_catalog.yaml \
  --source waxal_ug_asr \
  --language tw \
  --usage production \
  --revision <commit-sha> \
  --output data/external/waxal-tw
```

Provider snapshots are raw inputs. Provider-specific adapters map reviewed upstream schemas into our corpus schema. Unknown-speaker data is never assigned a fabricated speaker ID merely to satisfy split tooling.

## ASR experiments and benchmarks

Candidate architectures are versioned in `training/configs/asr_candidates.yaml`. Whisper-small is the first production baseline and Whisper-medium is the immediate capacity comparison. Alternative families remain visible but fail closed until their tokenizer/trainer prerequisites are implemented.

Create a reproducible experiment plan:

```bash
python -m training.experiments.experiment_plan \
  --language tw \
  --dataset-version artifacts/tw/dataset_version.json \
  --candidate whisper-small \
  --output artifacts/tw/experiments/whisper-small.json
```

Run locally or render the exact same frozen plan for a cluster:

```bash
python -m training.experiments.runner --plan artifacts/tw/experiments/whisper-small.json --backend local
python -m training.experiments.runner --plan artifacts/tw/experiments/whisper-small.json --backend slurm --output artifacts/tw/job.sbatch
python -m training.experiments.runner --plan artifacts/tw/experiments/whisper-small.json --backend kubernetes --output artifacts/tw/job.json
```

Speech evaluation uses WER/CER plus diagnostic slices and serving performance—not SWE-bench. The normalized benchmark includes dialect, speaker, noise, and code-switch slices, p50/p95 latency, and real-time factor.

Promotion thresholds live in `training/configs/benchmark_gates.yaml`.

## TTS experiments

NeMo FastPitch + HiFi-GAN remains the transparent baseline. Production training is blocked until the target-language grapheme/tokenizer policy is reviewed:

```bash
python -m training.tts.preflight \
  --profile training/configs/languages/tw.yaml \
  --artifacts artifacts/tw
```

VoxCPM2 is an advanced adaptation candidate. Freeze a data-lineage-bound LoRA experiment:

```bash
python -m training.tts.build_voxcpm_experiment \
  --language tw \
  --dataset-version artifacts/tw/dataset_version.json \
  --mode lora \
  --output experiments/voxcpm2-tw.json
```

The runtime refuses to treat an unadapted base checkpoint as proof of Twi/Ga/Ewe/Hausa competence. Candidate TTS models require blind native-speaker ratings for naturalness, pronunciation, intelligibility, and speaker similarity.

## Model registration, promotion, rollback

Training outputs keep model cards tied to exact corpus fingerprints and metrics. Deployment pointers are kept separate from immutable artifacts:

```text
candidate -> staging -> production -> retired
```

Promote or rollback registered artifacts with:

```bash
python -m training.deploy.promote \
  --root models \
  --task asr \
  --language tw \
  --stage production \
  --model-id <registered-model-id>

python -m training.deploy.promote \
  --root models \
  --task asr \
  --language tw \
  --stage production \
  --rollback
```

Pointer replacement is atomic; rollback never mutates a checkpoint directory.

## Realtime conversation protocol

`/v1/conversation` accepts PCM16 microphone frames. Processing runs in a cancellable child task so a fresh `start` message can barge into an older turn. Cancellation prevents stale output from being delivered; hard cancellation of already-running GPU kernels remains engine-specific.

Protocol details: `docs/WEBSOCKET_PROTOCOL.md`.

## TypeScript / React Native transport SDK

```bash
cd sdk/typescript
npm install --ignore-scripts
npm run build
```

The SDK contains typed HTTP helpers and an interruptible conversation WebSocket client. React Native microphone capture remains an app/platform adapter; the transport layer does not pretend browser `Blob` APIs are universal.

## Validation

```bash
python -m compileall -q services training tests examples
ruff check services training tests examples
pytest -q

cd sdk/typescript
npm install --ignore-scripts
npm run build
```

GitHub Actions runs the Python compile/Ruff/pytest gate and strict TypeScript build on every push and pull request.

Engineering decisions: `docs/LEAD_ENGINEER_DECISIONS.md`  
Current backlog: `docs/EXECUTION_BACKLOG.md`
