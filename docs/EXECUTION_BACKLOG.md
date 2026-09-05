# Execution backlog

A task is marked done only when its implementation and a repeatable verification gate exist.

## DONE — runtime and data factory

- [x] Canonical Twi (`tw`), Ga (`gaa`), Ewe (`ee`), Hausa (`ha`) profiles and aliases.
- [x] Orthography-preserving runtime normalization with no invented language rules.
- [x] Language-specific ASR/TTS routing with fail-closed missing checkpoints.
- [x] HTTP + interruptible WebSocket voice runtime.
- [x] Short-clip and long-recording first-party corpus intake.
- [x] Bounded audio decoding, normalization, segmentation, and SHA-256 deduplication.
- [x] SQLite corpus ledger, provenance, append-only audit, and two-independent-reviewer workflow.
- [x] Team-recording inventory importer with multi-speaker quarantine.
- [x] External-source catalog with production/evaluation/research boundaries.
- [x] Approved public bootstrap mix for Twi/Ga/Ewe/Hausa ASR/TTS.
- [x] Separate train/evaluation source roles; evaluation metadata never enters trainer metadata.
- [x] Pinned/revision-locked Hugging Face acquisition with immutable `SOURCE_RECEIPT.json`.
- [x] Streaming subset/split acquisition so giant repositories are not cloned wholesale.
- [x] Truthful governance path: first-party consent OR approved licensed-external provenance.
- [x] Public corpora lacking speaker IDs forced to `training_only=true`.
- [x] Speaker-disjoint corpus compiler, fingerprints, source-license/revision lineage, quality reports, and grapheme inventory.
- [x] Offline bootstrap-plan validation in normal GitHub CI.
- [x] Live Hugging Face smoke workflow for every language/task training combination.
- [x] Whisper experiment planning and local/Slurm/Kubernetes execution rendering.
- [x] WER/CER plus dialect/speaker/noise/code-switch, latency, and real-time-factor reporting.
- [x] Immutable model registration, atomic staging/production pointers, and rollback.
- [x] NeMo FastPitch + HiFi-GAN guarded baseline launchers.
- [x] VoxCPM2 runtime adapter and immutable adaptation-plan builder.
- [x] Blind native-speaker TTS listening evaluation aggregation.
- [x] Strict TypeScript HTTP/WebSocket SDK.
- [x] GitHub compile/Ruff/pytest/SDK CI.

## IN MOTION — public corpus-v0

- [ ] Live acquisition smoke must pass against the current upstream schemas for all non-gated training sources.
- [ ] Resolve-and-lock revisions for Twi Words 400K, Ga 90K, Ewe navigation, and Hausa Common-Voice-derived corpora.
- [ ] Execute full Twi public ASR/TTS corpus-v0 on storage compute.
- [ ] Execute full Ga public ASR/TTS corpus-v0; add gated CDLI eval after access is granted.
- [ ] Execute full Ewe public ASR/TTS corpus-v0.
- [ ] Execute full Hausa public ASR/TTS corpus-v0.
- [ ] Inspect `rejected.json`, hours, duplicate rate, clipping/noise flags, and observed grapheme inventories before GPU training.

## NEXT — model experiments

- [ ] Freeze identical Whisper-small baseline plans for `tw`, `gaa`, `ee`, and `ha`.
- [ ] Train/evaluate Whisper-small against held-out evaluation sources.
- [ ] Run Whisper-medium only where quality gains justify VRAM/latency cost.
- [ ] Approve observed grapheme inventories and spoken-normalization golden sets.
- [ ] Implement W2v-BERT training only after reviewed target vocabularies exist.
- [ ] Train NeMo TTS baselines on reviewed corpus-v0 TTS data.
- [ ] Run VoxCPM2 language-adaptation experiments.
- [ ] Collect blinded native-speaker pronunciation/naturalness/intelligibility/speaker-similarity ratings.
- [ ] Promote only candidates passing benchmark gates and engineering review.

## LATER — first-party corpus-v1

- [ ] Import consented team recordings as a separate provenance layer.
- [ ] Diarize quarantined multi-speaker meetings.
- [ ] Generate machine draft transcripts.
- [ ] Complete two independent human transcript-review passes.
- [ ] Freeze corpus-v1 and measure improvement over immutable public corpus-v0.

## HUMAN EVIDENCE — cannot be replaced by generated labels

- [ ] Retain first-party consent/source records outside model manifests.
- [ ] Provide speaker IDs, language, dialect, and multi-speaker status for first-party recordings.
- [ ] Native review of Twi/Ga/Ewe/Hausa transcript and normalization gold sets.
- [ ] Native-speaker TTS listening judgments.
