# Execution backlog

Tasks are marked done only when source code and an automated or human-review gate exist.

## DONE — platform and data factory

- [x] Canonical Twi (`tw`), Ga (`gaa`), Ewe (`ee`), Hausa (`ha`) profiles and aliases.
- [x] Orthography-preserving runtime normalization with no invented language rules.
- [x] Language-specific ASR/TTS routing contracts with fail-closed missing checkpoints.
- [x] Short-clip and long-recording corpus intake.
- [x] Bounded decoding/normalization and deterministic single-speaker segmentation.
- [x] SQLite corpus ledger, SHA-256 deduplication, provenance, and append-only review audit.
- [x] Machine draft -> reviewer 1 -> reviewer 2 -> approved workflow.
- [x] Separate reviewer identities enforced before approval.
- [x] Team-recording inventory importer; multi-speaker sources are quarantined for diarization.
- [x] External-source catalog with production/evaluation/research boundaries.
- [x] Provider adapters that refuse to fabricate missing speaker IDs.
- [x] Speaker-disjoint split compiler, corpus fingerprints, quality reports, and model lineage.
- [x] Whisper experiment-plan generation and local/Slurm/Kubernetes execution rendering.
- [x] WER/CER, speaker/dialect/noise/code-switch slices, latency, and real-time-factor reporting.
- [x] Atomic staging/production model pointers and rollback.
- [x] NeMo FastPitch + HiFi-GAN guarded baseline launchers.
- [x] VoxCPM2 runtime adapter and immutable adaptation-plan builder.
- [x] Blind native-listening aggregation for TTS candidates.
- [x] Strict TypeScript HTTP/WebSocket SDK.
- [x] GitHub CI for Python compile/Ruff/pytest and TypeScript compilation.

## NEXT — requires real data or compute

- [ ] Import the first consented Twi/Ga/Ewe/Hausa recording batch.
- [ ] Diarize quarantined multi-speaker meetings before speaker-conditioned use.
- [ ] Complete two independent transcript-review passes for the first frozen corpus versions.
- [ ] Approve observed grapheme inventories and language-specific spoken-normalization golden sets.
- [ ] Run Whisper-small baselines on identical held-out sets for all four languages.
- [ ] Run Whisper-medium comparisons where baseline error/latency justify the cost.
- [ ] Implement the alternative W2v-BERT trainer after reviewed vocabulary policy exists.
- [ ] Run NeMo and VoxCPM2 TTS adaptation experiments on reviewed corpora.
- [ ] Collect blinded native-speaker pronunciation/naturalness/intelligibility ratings.
- [ ] Promote only candidates passing benchmark gates and engineering review.

## HUMAN EVIDENCE — cannot be replaced by generated labels

- [ ] Consent/source records retained outside the training manifests.
- [ ] Speaker IDs, language, dialect, and multi-speaker status for first-party recordings.
- [ ] Native review of Twi/Ga/Ewe/Hausa transcript and normalization gold sets.
- [ ] Native TTS listening judgments.
