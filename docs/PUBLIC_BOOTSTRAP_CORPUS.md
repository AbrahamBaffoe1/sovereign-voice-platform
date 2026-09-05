# Public bootstrap corpus v0

This runbook freezes the public-data baseline for Twi, Ga, Ewe, and Hausa before first-party team recordings are added.

## Design rules

1. Training and evaluation sources are different roles. An evaluation source never enters `metadata.csv` used to build the training artifact.
2. Every Hugging Face source is locked to a full commit SHA before rows are processed. Sources without a version-controlled SHA in the catalog resolve HEAD exactly once into `data/bootstrap/locks/*.json`; later runs reuse that lock unless `--refresh-lock` is explicit.
3. Large repositories are read only through their configured subset/split. We do not snapshot all of WAXAL. WAXAL Ewe ASR uses a catalog-declared raw Parquet path because the upstream physical shard contains an undeclared pandas index column that breaks repository-level feature casting.
4. Audio normalization is task-specific and comes from each version-controlled language profile: ASR is 16 kHz mono PCM16 WAV; current custom-language TTS profiles are 22.05 kHz mono PCM16 WAV.
5. External rows preserve their actual governance basis. `consent_attested=false` is expected for public corpora; `governance_approved=true`, source license, dataset revision, repo/config/split, and upstream-validation status prove why the row is allowed.
6. Public corpora without trustworthy speaker IDs are `training_only=true` for normal split generation. An explicitly reserved evaluation corpus is compiled independently with `--fixed-split test`, so source hints can never repartition it into train.
7. After both artifacts are frozen, normalized audio SHA-256 values are compared across training and evaluation. Any exact waveform overlap fails the build and writes `exact_audio_leakage.json`.
8. Completed source acquisition is resumable only when revision, task sample rate, requested sample limit, metadata hash, and referenced audio files still match. A one-row smoke receipt can never satisfy a full-corpus build.
9. Rejected upstream rows are journaled per source in `rejected.jsonl`; the strict compiler also emits its own rejection/quality artifacts.
10. First-party recordings remain a separate source layer and can be added to corpus-v1 later without losing the corpus-v0 baseline.

## Approved bootstrap mix

### Twi

ASR training: WAXAL Akan train + Twi Words 400K lexical augmentation.  
ASR evaluation: WAXAL Akan v2 test.  
TTS training: WAXAL Twi TTS + Twi Words 400K pronunciation augmentation.

### Ga

ASR/TTS training: GhanaNLP Ga 90K word-level corpus.  
ASR evaluation: CDLI Ga standard speech, gated and kept held out.

### Ewe

ASR training: WAXAL Ewe train.  
ASR evaluation: WAXAL Ewe test.  
TTS training: WAXAL Ewe TTS.  
The Ghana Open Data Ewe navigation corpus is currently `CC-BY-NC-4.0`; it remains research-only and is excluded from production corpus-v0.

### Hausa

ASR training: CC-BY Common-Voice-derived Hausa bootstrap corpus.  
ASR evaluation: small dialect-aware Dialectra corpus.  
TTS training: WAXAL Hausa TTS + BibleTTS Hausa.

CC-BY-SA sources stay explicitly labeled in every row/receipt so attribution/share-alike obligations are not lost downstream.

## Install the data toolchain

```bash
pip install -e '.[data]'
```

## Preview without downloading

The dry run reports the exact source revisions, roles, task sample rates, and the filesystem capacity that would back the requested output roots:

```bash
python -m training.data.bootstrap \
  --language all \
  --task both \
  --include-eval \
  --dry-run
```

## Live smoke acquisition

GitHub Actions runs a one-real-row matrix against every production language/task combination whenever acquisition/catalog code changes. For a larger local/provider smoke test:

```bash
python -m training.data.bootstrap \
  --language tw \
  --task asr \
  --max-samples 25
```

`--max-samples` is part of the durable source receipt, so a smoke acquisition is never reused as a completed full source.

## Full bootstrap on persistent storage

Do not use ephemeral GitHub Actions storage for the full corpora. Point both roots at persistent volumes and choose a free-space threshold appropriate for the volume you provisioned:

```bash
python -m training.data.bootstrap \
  --language all \
  --task both \
  --include-eval \
  --data-root /mnt/voice-data/bootstrap \
  --artifacts-root /mnt/voice-artifacts/bootstrap \
  --min-free-gb 100
```

For a release gate where every planned benchmark must be available, use `--require-eval` (it implies `--include-eval`):

```bash
python -m training.data.bootstrap \
  --language all \
  --task asr \
  --require-eval \
  --data-root /mnt/voice-data/bootstrap \
  --artifacts-root /mnt/voice-artifacts/bootstrap \
  --min-free-gb 100
```

For gated CDLI data, accept the upstream dataset terms in the Hugging Face account and export `HF_TOKEN` before the Ga run. Without `--require-eval`, unavailable optional evaluation sources are recorded in `optional_failures` without contaminating or blocking the training corpus.

Use `--force-reacquire` only when normalized source files must be rebuilt despite a matching valid receipt. Use `--refresh-lock` only when intentionally changing an unpinned upstream revision; doing so changes reproducibility lineage and should be reviewed like a data-version change.

## Outputs

Each source produces:

```text
data/bootstrap/<language>/<task>/<role>/<source>/
  audio/*.wav
  metadata.csv
  rejected.jsonl
  SOURCE_RECEIPT.json
```

The task directory also contains:

```text
metadata.csv                 # training-role sources only
evaluation_metadata.csv      # evaluation-role sources only, when requested/available
ACQUISITION_SUMMARY.json
```

Training compilation emits:

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

Reserved evaluation compilation emits a physically separate artifact:

```text
artifacts/bootstrap/<language>/<task>/corpus-v0-eval/
  train.json                 # empty by construction
  validation.json            # empty by construction
  test.json                  # all accepted reserved benchmark rows
  audit.jsonl
  rejected.json
  inventory.json
  quality_report.json
  dataset_version.json
```

When both exist, the builder also writes:

```text
artifacts/bootstrap/<language>/<task>/exact_audio_leakage.json
artifacts/bootstrap/BUILD_REPORT.json
```

`BUILD_REPORT.json` records acquisition summaries, frozen dataset fingerprints, sample rates, evaluation state, leakage proof, and pre/post filesystem capacity. That file is the handoff boundary to baseline model training.

## Next model boundary

Only after a full corpus-v0 build is frozen and reviewed do we start Whisper-small baseline experiments. The trainer consumes `corpus-v0/train.json`, `validation.json`, and `test.json`; the independently frozen `corpus-v0-eval/test.json` remains an external benchmark and is never used for optimizer updates or model selection.
