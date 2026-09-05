# Intern Tasks

These TODOs are intentionally limited to work that should be reviewed before merge. Core runtime architecture and model interfaces are already implemented.

## P0 — Twi corpus and language frontend

- [ ] Collect/organize recordings with documented speaker permission and exact transcripts.
- [ ] Run `training.prepare_dataset`; manually inspect every rejection category.
- [ ] Have two native Twi speakers review at least 500 randomly sampled transcript/audio pairs.
- [ ] Review `training/configs/languages/twi.yaml` against the actual corpus character inventory.
- [ ] Build a golden text-normalization set covering numbers, dates, time, Ghana cedi/pesewa amounts, abbreviations, names, loan words and code-switching.
- [ ] Implement those reviewed rules in `services/api/app/normalization/twi.py`; every rule needs a golden test.
- [ ] Decide grapheme-vs-phoneme FastPitch frontend based on pronunciation error experiments. Do not make this decision from convenience alone.
- [ ] If phoneme-based: create the pronunciation lexicon/G2P mapping with linguist review and explicit OOV policy.

## P1 — Model evaluation

- [ ] ASR: create a frozen speaker-diverse test set and report WER plus named-entity/number error slices.
- [ ] TTS: run blinded native-speaker MOS/pronunciation evaluation; record failure categories, not only average MOS.
- [ ] Add code-switch test sets for Twi-English utterances.
- [ ] Record RTF, first-response latency, peak VRAM and sustained throughput for each checkpoint.

## P1 — Client SDK

- [ ] Implement React Native microphone capture producing mono PCM16 frames.
- [ ] Add reconnect/backpressure logic based on `examples/ws_client.py` and the protocol doc.
- [ ] Never buffer unbounded microphone data on device.

## P2 — Production

- [ ] Add CI that runs compile, tests, Ruff and dependency vulnerability scans.
- [ ] Add reverse-proxy TLS config for the deployment environment.
- [ ] Add rate limiting and authenticated tenant identity before exposing voice enrollment publicly.
- [ ] Add immutable model version metadata and promotion scripts.
