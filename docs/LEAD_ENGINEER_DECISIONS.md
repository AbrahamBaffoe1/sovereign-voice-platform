# Lead engineer decisions

## Target languages

The first custom-language release train is Twi (`tw`), Ga (`gaa`), Ewe (`ee`), and Hausa (`ha`). Every language uses the same platform contracts but owns separate data versions, evaluation slices, and production model pointers.

## Data governance

First-party consented recordings are the primary corpus. External sources are admitted through `training/configs/source_catalog.yaml` and classified as production, evaluation, or research-only. Unknown-speaker public corpora may augment training but must not enter speaker-disjoint validation/test sets. Machine transcripts are drafts only; two independent human review stages are required before an item becomes an approved training label.

## ASR

Whisper-small is the first reproducible baseline and Whisper-medium is the immediate capacity comparison. Alternative model families remain blocked until tokenizer/trainer prerequisites are implemented and reviewed. Promotion requires held-out WER/CER plus latency and real-time-factor reports; aggregate WER alone is insufficient.

## TTS

NeMo FastPitch + HiFi-GAN remains the transparent baseline. VoxCPM2 is an advanced adaptation candidate, but the generic base checkpoint is never treated as proof of Twi/Ga/Ewe/Hausa support. A language is ready only after an adapted checkpoint passes native-speaker pronunciation, naturalness, intelligibility, and serving-latency gates.

## Runtime

Model artifacts are immutable. Staging/production selection is done through atomic JSON pointers so rollback does not rewrite checkpoints. Long multi-speaker recordings are quarantined for diarization rather than silently assigning all speech to one person.

## Engineering quality

GitHub Actions is the clean-room gate. Python compilation, Ruff, pytest, and strict TypeScript compilation must pass before a sprint is considered published. Model-quality benchmarks are separate from software CI.
