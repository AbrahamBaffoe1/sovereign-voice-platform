# First-party Twi recordings: intake review

Source repository commit: `9a3f768aafae0bbb1391df1daae14f1732e5b0aa`.
The user confirmed that all recordings are Twi and the same person speaks in each.
Use the stable, anonymous speaker ID `team_tw_speaker_001`; dialect remains unknown.
The user attests ownership and intended training use. This statement is recorded;
speaker-specific consent evidence has not yet been entered in the importer field.

## Measured audio checks

- 21 original Opus files; all decoded successfully with no FFmpeg error output.
- 20 unique files, totaling 274.410 seconds (4 minutes 34.410 seconds).
- `21.opus` is byte-identical to `17.opus` (SHA-256 confirmed). Originals are preserved;
  only `17.opus` is included in the intake and transcript review lists.
- Unique recording durations: 9.6535 to 20.2335 seconds.
- All streams decode as mono at 48 kHz. This is a decoder rate, not a claim about
  the original recording bandwidth or microphone quality.
- `10.opus` is 20.2335 seconds, above the current corpus maximum of 20 seconds.
  Split it at a reviewed utterance boundary and update its matching transcript.
- 19 of 20 unique recordings contain decoded floating-point samples above full scale.
  The highest peak is +3.19 dBFS (`1.opus`). Apply sufficient attenuation before
  PCM16 conversion and check peaks again after resampling. These measurements alone
  do not prove audible clipping or identify when distortion may have occurred.
- Quiet-frame percentages in `audio_audit.csv` use 20 ms RMS below -50 dBFS;
  these are signal measurements, not speech/non-speech classifications.

## Review files and next steps

`datasets/team_recordings_inventory.csv` contains the 20 unique source recordings.
Its paths are relative to the repository root. `transcript_review.csv` contains
one blank, unreviewed transcript row per unique recording. It is a review worksheet,
not an approved training manifest. No transcript has been generated or verified.

1. Listen to each recording and enter the exact Twi text; have a fluent reviewer
   check it against the audio before setting `transcript_reviewed=true`.
2. Record speaker training permission in the inventory's `consent_attested` field
   when supported by collection evidence. The existing importer rejects blank values.
3. Check speech quality, segment at utterance boundaries as needed, and produce
   separate mono PCM16 derivatives for ASR (16 kHz) and the planned TTS corpus
   (22.05 kHz). Keep source Opus files unchanged. Resampling cannot restore
   information already lost to compression.
4. Keep this speaker grouped for corpus splitting. These recordings cannot alone
   provide a speaker-disjoint train/evaluation split. Use independently held-out
   speakers for the ASR benchmark. Any same-speaker TTS holdout must be described
   explicitly as such and keep related segments from one source together.

This batch remains pending review. It has not been imported into an approved
corpus, added to the running Twi experiment, or used to start another GPU run.
