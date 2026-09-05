# Real training execution

This is the operator handoff for moving corpus-v0 and Whisper-small baselines off ephemeral CI and onto a persistent GPU machine.

## Execution contract

Use a Linux x64 host with:

- an NVIDIA GPU visible to both `nvidia-smi` and `torch.cuda`;
- Python 3.11;
- a persistent filesystem mounted outside the Git checkout;
- enough free disk for source audio, normalized WAVs, Hugging Face caches and checkpoints;
- outbound access to approved Hugging Face sources;
- `HF_TOKEN` when a gated source such as the Ga held-out corpus is enabled.

The default persistent root is `/srv/sovereign-voice`. Override it with `VOICE_EXECUTION_ROOT` or `--workspace`.

The execution code rejects `/tmp`, `$RUNNER_TEMP`, and `$GITHUB_WORKSPACE` as production storage. Git checkout cleanup must never be able to erase a downloaded corpus or a multi-hour checkpoint.

## Persistent layout

```text
/srv/sovereign-voice/
  data/bootstrap/                 acquired + normalized source audio
  artifacts/bootstrap/            frozen corpus-v0 and held-out artifacts
  artifacts/experiments/asr/      Hugging Face + CTranslate2 ASR runs
  artifacts/tts-readiness/        evidence packets for tokenizer/frontend review
  cache/                          Hugging Face, Torch and XDG caches
  logs/real-execution.log         durable combined subprocess log
  state/EXECUTION_ENVIRONMENT.json
  state/REAL_EXECUTION.json
  state/execution.lock
  venv/                           persistent Python environment when launched by Actions
```

## First machine check

Install the project training dependencies in the environment that will own the GPU job:

```bash
python -m pip install -e '.[data,training-asr,asr,training]'
```

Then run one language first. The 150 GiB gate is a safety floor, not a corpus-size claim; set it to the headroom policy of the actual disk.

```bash
python -m training.execution.run_pipeline \
  --workspace /srv/sovereign-voice \
  --language tw \
  --min-free-gb 150
```

The pipeline executes:

```text
source acquisition/resume
  -> corpus-v0 freeze
  -> independently frozen held-out benchmark when available
  -> exact-audio leakage gate
  -> TTS readiness evidence generation
  -> Whisper-small training/resume
  -> CTranslate2 export
  -> internal and external ASR evaluation
```

TTS readiness does **not** train FastPitch yet. The current profiles intentionally block production TTS training until a native-speaker/linguist-reviewed frontend/tokenizer/grapheme policy is committed. Observed corpus characters are evidence, not automatic pronunciation rules.

## Resume behavior

Re-run the same command after a machine/process interruption.

ASR experiment identity is tied to:

- language;
- base model;
- language-profile hash;
- frozen corpus version hash/fingerprint;
- checkpoint-selection policy;
- training hyperparameters.

If those change, resume fails rather than mixing incompatible optimizer/model state. If they match, the trainer selects the highest `checkpoint-*` directory automatically.

A completed Hugging Face model is not retrained merely because export/evaluation failed. CTranslate2 output is derived data and can be deleted/rebuilt safely; the authoritative HF checkpoint is preserved.

## GitHub Actions

`.github/workflows/real-execution.yml` targets a self-hosted Linux x64 runner. The runner must have the persistent disk mounted at `VOICE_EXECUTION_ROOT` and an NVIDIA GPU.

Recommended repository variable:

```text
VOICE_EXECUTION_ROOT=/srv/sovereign-voice
```

Optional repository secret for gated Hugging Face data:

```text
HF_TOKEN=<token with accepted-dataset access>
```

The workflow also supports manual language selection, disk headroom, external-benchmark strictness, and source reacquisition.

The workflow does not upload corpora or checkpoints back to GitHub Actions artifacts. They remain on the persistent workspace and are identified by the JSON lineage files in that workspace.

## Release gate

An ASR run is a deployment candidate only when:

1. corpus-v0 is non-empty and fingerprinted;
2. training/evaluation exact-audio leakage is zero;
3. the run result is `completed`;
4. the final model's recorded corpus/profile lineage matches the frozen artifact;
5. CTranslate2 export completed;
6. required held-out evaluation is present and reviewed;
7. WER/CER and dialect slices are reviewed against the acceptance criteria for that language.

Do not set `--require-external-eval` for all languages until every gated benchmark is accessible. The normal execution path will still evaluate any external benchmark that was successfully frozen.
