# Real training execution

This is the operator handoff for moving corpus-v0 and Whisper-small baselines off ephemeral CI and onto persistent GPU compute.

## Execution contract

Use a Linux x64 host with:

- an NVIDIA GPU visible to both `nvidia-smi` and `torch.cuda`;
- Python 3.11;
- a persistent filesystem mounted outside the Git checkout;
- enough free disk for source audio, normalized WAVs, model caches and checkpoints;
- outbound access to approved Hugging Face sources;
- `HF_TOKEN` when an accepted/gated source is enabled.

The execution code rejects `/tmp`, `$RUNNER_TEMP`, and `$GITHUB_WORKSPACE` as production storage. Checkout cleanup must never be able to erase a downloaded corpus or a multi-hour checkpoint.

## Direct GPU host — preferred for this public repository

On a GPU VM/pod with a persistent volume mounted at `/workspace`, clone the repository onto the machine and run:

```bash
export VOICE_EXECUTION_ROOT=/workspace/sovereign-voice
export EXEC_LANGUAGE=tw
export MIN_FREE_GB=150
./scripts/run_real_training.sh
```

The 150 GiB value is an operator safety floor, not a claim about final corpus size. Adjust it to the actual disk/headroom policy.

The launcher keeps its virtual environment and dependency snapshot on the persistent volume. Recreating the cloud machine therefore does not force corpus acquisition or ASR training to restart from zero.

The production launcher also applies `constraints/training-cu124.txt` so a future PyTorch release
cannot silently move the RunPod environment to an incompatible CUDA major version. Operators testing
another reviewed CUDA stack must set `PIP_CONSTRAINT` and `EXPECTED_TORCH_CUDA` together.

## Persistent layout

```text
<VOICE_EXECUTION_ROOT>/
  data/bootstrap/                 acquired + normalized source audio
  artifacts/bootstrap/            frozen corpus-v0 and held-out artifacts
  artifacts/experiments/asr/      Hugging Face + CTranslate2 ASR runs
  artifacts/tts-readiness/        evidence packets for tokenizer/frontend review
  cache/                          Hugging Face, Torch and XDG caches
  logs/real-execution.log         durable combined subprocess log
  state/EXECUTION_ENVIRONMENT.json
  state/REAL_EXECUTION.json
  state/pip-freeze.txt
  state/pip-constraint.txt
  state/execution.lock
  venv/
```

## Pipeline

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

A completed Hugging Face model is not retrained merely because export/evaluation failed. CTranslate2 output is derived data and can be rebuilt; the authoritative HF checkpoint is preserved.

## GitHub Actions fallback

`.github/workflows/real-execution.yml` is an alternative when a trusted self-hosted runner is appropriate. Because this repository is public, do not attach a general-purpose machine. Use a dedicated GPU runner, keep the workflow main-only, and give the runner the custom `sovereign-gpu` label.

The runner needs the persistent disk mounted at `VOICE_EXECUTION_ROOT`. Recommended repository variable:

```text
VOICE_EXECUTION_ROOT=/srv/sovereign-voice
```

Optional repository secret for gated Hugging Face data:

```text
HF_TOKEN=<token with accepted-dataset access>
```

The workflow does not upload corpora or checkpoints as GitHub Actions artifacts. The persistent workspace is the source of truth, with source revision and environment lineage recorded beside the model.

## Release gate

An ASR run is a deployment candidate only when:

1. corpus-v0 is non-empty and fingerprinted;
2. training/evaluation exact-audio leakage is zero;
3. the run result is `completed`;
4. final model lineage matches the frozen corpus/profile;
5. CTranslate2 export completed;
6. required held-out evaluation is present and reviewed;
7. WER/CER and dialect slices meet the acceptance criteria for that language.

Do not require every external benchmark globally until every gated source is accessible. The normal execution path still evaluates each held-out benchmark that was successfully frozen.
