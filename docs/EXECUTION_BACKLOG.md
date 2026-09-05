# Execution backlog

Tasks move to DONE only when code, tests and a reviewable artifact exist.

## DONE — four-language training foundation

- [x] Canonical runtime support for Twi (`tw`), Ga (`gaa`), Ewe (`ee`) and Hausa (`ha`).
- [x] Human-friendly aliases without duplicate model caches.
- [x] Orthography-preserving normalizers for all four languages.
- [x] Version-controlled training profiles for all four languages.
- [x] Per-language custom ASR routing with no silent generic-model fallback.
- [x] Per-language custom NeMo TTS routing with checkpoint readiness checks.
- [x] Corpus governance fields: speaker, dialect, source ID, consent, transcript review.
- [x] Speaker-disjoint dataset splits.
- [x] Audit manifests, grapheme inventories and corpus quality reports.
- [x] Whisper token-free experiment mode for unreviewed decoder-token strategies.
- [x] TTS preflight blocking unreviewed tokenizer/grapheme policies.
- [x] Stable dataset version IDs/fingerprints.
- [x] Candidate `model_card.json` metadata with artifact hashes and dataset lineage.

## NEXT — software-only work

- [ ] Corpus-ingestion API with reviewer workflow.
- [ ] Transcript states: machine draft -> reviewer 1 -> reviewer 2 -> approved.
- [ ] ASR experiment runner for multiple base-model sizes.
- [ ] WER/CER slices by dialect, speaker, noise and code-switching.
- [ ] Model promotion workflow: candidate -> staging -> production -> retired.
- [ ] Atomic CTranslate2 export/deployment and rollback.
- [ ] TTS grapheme-vs-phoneme experiment registry.
- [ ] Native listening/MOS evaluation workflow.
- [ ] Streaming VAD/endpointing, cancellation and barge-in.
- [ ] TypeScript and React Native SDKs.

## BLOCKED — needs language/data experts

- [ ] Twi reviewed grapheme inventory and spoken normalization golden set.
- [ ] Ga reviewed grapheme inventory and spoken normalization golden set.
- [ ] Ewe reviewed grapheme inventory and spoken normalization golden set.
- [ ] Hausa reviewed Boko inventory and Ajami/Boko product decision.
- [ ] Licensed/consented multi-speaker recordings.
- [ ] Human-reviewed transcripts and dialect labels.
- [ ] Native-speaker TTS listening evaluation.
