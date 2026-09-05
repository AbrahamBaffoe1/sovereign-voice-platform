# Public bootstrap corpus v0

This runbook freezes the public-data baseline for Twi, Ga, Ewe, and Hausa before first-party team recordings are added.

## Design rules

1. Training and evaluation sources are different roles. An evaluation source never enters `metadata.csv` used by the trainer.
2. Every Hugging Face source is locked to a full commit SHA before rows are processed. Sources without a version-controlled SHA in the catalog resolve HEAD exactly once into `data/bootstrap/locks/*.json`; later runs reuse that lock unless `--refresh-lock` is explicit.
3. Large repositories are streamed by configured dataset subset/split. We do not snapshot all of WAXAL.
4. Every clip is normalized to mono PCM16 WAV at 16 kHz before the strict corpus compiler sees it.
5. External rows preserve their actual governance basis. `consent_attested=false` is expected for public corpora; `governance_approved=true`, the source license, dataset revision, repo/config/split, and upstream-validation status prove why the row is allowed.
6. Public corpora without trustworthy speaker IDs are `training_only=true`. They cannot populate validation/test through our compiler.
7. First-party recordings remain a separate source layer and can be added to corpus-v1 later without losing the corpus-v0 baseline.

## Approved bootstrap mix

### Twi

ASR training: WAXAL Akan train + Twi Words 400K lexical augmentation.  
ASR evaluation: WAXAL Akan v2 test.  
TTS training: WAXAL Twi TTS + Twi Words 400K pronunciation augmentation.

### Ga

ASR/TTS training: GhanaNLP Ga 90K word-level corpus.  
ASR evaluation: CDLI Ga standard speech, gated and kept held out.

### Ewe

ASR training: WAXAL Ewe train + Ewe navigation speech.  
ASR evaluation: WAXAL Ewe test.  
TTS training: WAXAL Ewe TTS + Ewe navigation speech.

### Hausa

ASR training: CC-BY Common-Voice-derived Hausa bootstrap corpus.  
ASR evaluation: small dialect-aware Dialectra corpus.  
TTS training: WAXAL Hausa TTS + BibleTTS Hausa.

CC-BY-SA sources stay explicitly labeled in every row/receipt so attribution/share-alike obligations are not lost downstream.

## Preview without downloading

```bash
pip install -e '.[data]'
python -m training.data.bootstrap --language all --task both --include-eval --dry-run
```

## Smoke acquisition

Before allocating large disk/GPU resources, acquire 25 rows from each planned training source:

```bash
python -m training.data.bootstrap \
  --language tw \
  --task asr \
  --max-samples 25
```

The command writes normalized audio under `data/bootstrap/` and compiled artifacts under `artifacts/bootstrap/`; both are ignored by Git.

## Full bootstrap

Run one language at a time so storage, provider access, and receipts can be inspected between jobs:

```bash
python -m training.data.bootstrap --language tw  --task both --include-eval
python -m training.data.bootstrap --language gaa --task both --include-eval
python -m training.data.bootstrap --language ee  --task both --include-eval
python -m training.data.bootstrap --language ha  --task both --include-eval
```

For gated CDLI data, accept the dataset terms in the upstream account and export `HF_TOKEN` before the Ga run. The source is optional, so lack of access is recorded as an optional failure rather than contaminating or blocking the Ga training corpus.

## Outputs

Each source produces:

```text
data/bootstrap/<language>/<task>/<role>/<source>/
  audio/*.wav
  metadata.csv
  SOURCE_RECEIPT.json
```

The task directory also contains:

```text
metadata.csv                 # training-role sources only
evaluation_metadata.csv      # evaluation-role sources only
ACQUISITION_SUMMARY.json
```

Compilation then emits `artifacts/bootstrap/<language>/<task>/corpus-v0/` with train/validation/test manifests, the richer `audit.jsonl`, rejected rows, grapheme inventory, quality report, and `dataset_version.json` fingerprint.
