# Sovereign Voice Platform

Self-hosted ASR -> optional local LLM -> TTS platform with first-class training infrastructure for **Twi (`tw`)**, **Ga (`gaa`)**, **Ewe (`ee`)**, and **Hausa (`ha`)**.

The repository contains source code, dataset governance metadata, training/evaluation tooling, model deployment pointers, and client SDKs. **Audio corpora and model weights are intentionally not committed to Git.**

## Current engineering status

Implemented:

- Faster-Whisper runtime and language-specific promoted ASR routes.
- Chatterbox, NeMo FastPitch/HiFi-GAN, and adapted VoxCPM2 TTS routes.
- HTTP + interruptible WebSocket voice APIs.
- Twi/Ga/Ewe/Hausa orthography-preserving normalization.
- First-party recording ingestion, segmentation, deduplication, and two-person transcript review.
- Public-corpus catalog with production/evaluation/research license boundaries.
- Streaming Hugging Face acquisition with immutable revision locks and source receipts.
- Speaker-disjoint corpus compilation, fingerprints, grapheme inventory, and quality reports.
- Whisper experiment plans for local, Slurm, and Kubernetes execution.
- WER/CER slices, latency/RTF benchmarks, native-speaker TTS evaluation, model promotion, and rollback.
- Strict TypeScript HTTP/WebSocket SDK.
- GitHub CI plus a live public-dataset smoke workflow.

The remaining critical path is **real corpus execution + GPU training/evaluation**, not another application scaffold.

## Install

```bash
cp .env.example .env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[asr,dev]'
make run
```

Install only the extras needed for the job:

```bash
pip install -e '.[data]'            # public dataset acquisition
pip install -e '.[training-asr]'    # Whisper training
pip install -e '.[tts-chatterbox]'
pip install -e '.[tts-nemo]'
pip install -e '.[tts-voxcpm]'
```

## Public bootstrap corpus v0

The approved source mix is version controlled in:

```text
training/configs/source_catalog.yaml
training/configs/bootstrap_corpora.yaml
```

Training and evaluation sources are separate roles. Evaluation rows are never merged into the training metadata.

Current bootstrap plan:

```text
Twi ASR  : WAXAL Akan train + Twi Words 400K
Twi eval : WAXAL Akan v2 test
Twi TTS  : WAXAL Twi TTS + Twi Words 400K

Ga ASR   : GhanaNLP Ga 90K
Ga eval  : CDLI Ga standard speech (gated, optional)
Ga TTS   : GhanaNLP Ga 90K baseline

Ewe ASR  : WAXAL Ewe train + GhanaNLP navigation speech
Ewe eval : WAXAL Ewe test
Ewe TTS  : WAXAL Ewe TTS + navigation speech

Hausa ASR  : CC-BY Common-Voice-derived Hausa bootstrap corpus
Hausa eval : Dialectra dialect-aware Hausa diagnostic set
Hausa TTS  : WAXAL Hausa TTS + BibleTTS Hausa
```

Large repositories are **streamed by configured subset/split**. The acquisition code does not clone or snapshot the complete WAXAL repository.

### Validate the plan without downloading audio

```bash
python -m training.data.bootstrap \
  --language all \
  --task both \
  --include-eval \
  --dry-run
```

This command is also part of normal GitHub CI.

### Smoke-test real upstream rows

```bash
python -m training.data.acquire \
  --language tw \
  --task asr \
  --max-samples 1
```

`.github/workflows/bootstrap-smoke.yml` runs this live check for all eight language/task combinations whenever acquisition/catalog code changes. It uses public access by default and can use an optional repository `HF_TOKEN` secret.

### Build full public corpus-v0

Run one language at a time on a machine with adequate disk:

```bash
python -m training.data.bootstrap --language tw  --task both --include-eval
python -m training.data.bootstrap --language gaa --task both --include-eval
python -m training.data.bootstrap --language ee  --task both --include-eval
python -m training.data.bootstrap --language ha  --task both --include-eval
```

For the gated CDLI Ga evaluation set, accept the upstream terms and provide `HF_TOKEN`. It is optional; lack of access does not contaminate or block Ga training.

Every external source produces:

```text
data/bootstrap/<language>/<task>/<role>/<source>/
  audio/*.wav
  metadata.csv
  SOURCE_RECEIPT.json
```

Each receipt records the source repository, exact resolved revision, task/role, license, row counts, hours, and metadata SHA-256. Sources without a hard-coded revision resolve once to a full commit SHA and are then frozen under `data/bootstrap/locks/` until an explicit `--refresh-lock`.

Compiled artifacts are written to:

```text
artifacts/bootstrap/<language>/<task>/corpus-v0/
  train.json
  validation.json
  test.json
  audit.jsonl
  rejected.json
  inventory.json
  quality_report.json
  dataset_version.json
```

## First-party recordings: corpus-v1 and later

Public corpus-v0 remains immutable as the baseline. Team recordings are added as a distinct provenance layer later.

Fill:

```text
datasets/team_recordings_inventory.csv
```

then run:

```bash
python -m training.ingest.import_team_recordings \
  --inventory datasets/team_recordings_inventory.csv \
  --input-root /path/to/original-recordings \
  --corpus-root data/corpus
```

Single-speaker recordings are normalized and segmented. `multi_speaker=true` recordings are quarantined for diarization rather than assigned to a fake speaker.

Transcript state machine:

```text
machine_draft
    -> reviewer_1_complete
    -> reviewer_2_complete
    -> approved
```

Machine drafts may help reviewers, but they cannot approve themselves.

## ASR baseline training

Once `corpus-v0/dataset_version.json` exists, freeze a reproducible experiment:

```bash
python -m training.experiments.experiment_plan \
  --language tw \
  --dataset-version artifacts/bootstrap/tw/asr/corpus-v0/dataset_version.json \
  --candidate whisper-small \
  --output artifacts/bootstrap/tw/asr/whisper-small.json
```

Execute the exact same scientific plan locally or render it for a cluster:

```bash
python -m training.experiments.runner --plan artifacts/bootstrap/tw/asr/whisper-small.json --backend local
python -m training.experiments.runner --plan artifacts/bootstrap/tw/asr/whisper-small.json --backend slurm --output job.sbatch
python -m training.experiments.runner --plan artifacts/bootstrap/tw/asr/whisper-small.json --backend kubernetes --output job.json
```

The first comparison is Whisper-small across all four languages on held-out data. Whisper-medium is a second experiment only where accuracy/latency results justify it.

ASR regression metrics include WER, CER, dialect/speaker/noise/code-switch slices, p50/p95 latency, and real-time factor. SWE-bench is not a speech benchmark.

## TTS training

NeMo FastPitch + HiFi-GAN is the transparent baseline. VoxCPM2 is the advanced adaptation candidate.

TTS training remains fail-closed until the observed grapheme/tokenizer policy for a language is reviewed:

```bash
python -m training.tts.preflight \
  --profile training/configs/languages/tw.yaml \
  --artifacts artifacts/bootstrap/tw/tts/corpus-v0
```

Freeze a VoxCPM2 adaptation plan:

```bash
python -m training.tts.build_voxcpm_experiment \
  --language tw \
  --dataset-version artifacts/bootstrap/tw/tts/corpus-v0/dataset_version.json \
  --mode lora \
  --output experiments/voxcpm2-tw.json
```

A TTS candidate is not production-ready until native speakers rate pronunciation, intelligibility, naturalness, and speaker similarity and the serving benchmark passes.

## Model promotion and rollback

Model artifacts are immutable. Deployment changes only atomic pointers:

```text
candidate -> staging -> production -> retired
```

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

## Validation

```bash
python -m compileall -q services training tests examples
ruff check services training tests examples
pytest -q
python -m training.data.bootstrap --language all --task both --include-eval --dry-run

cd sdk/typescript
npm install --ignore-scripts
npm run build
```

See:

- `docs/PUBLIC_BOOTSTRAP_CORPUS.md` — public-data execution details.
- `docs/LEAD_ENGINEER_DECISIONS.md` — architecture/model decisions.
- `docs/EXECUTION_BACKLOG.md` — current execution state.
- `docs/WEBSOCKET_PROTOCOL.md` — conversation transport protocol.
