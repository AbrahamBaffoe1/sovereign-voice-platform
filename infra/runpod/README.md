# RunPod production training contract

Use a **Pod**, not Serverless, for corpus acquisition and model training. The workload is stateful, long-running, and checkpoint-resumable; a persistent mounted volume is the authoritative storage boundary.

## First production lane

Start with Twi only. Do not fan out to Ga/Ewe/Hausa until one complete lane has validated corpus acceptance, storage growth, GPU memory, checkpoint recovery, CTranslate2 export, and WER/CER reporting.

Recommended first host:

```text
GPU count:        1
VRAM class:       48 GB
Preferred order:  A40 / RTX A6000 / RTX 6000 Ada / L40S
CPU/RAM:          use the Pod allocation paired with the selected GPU
Persistent disk:  >= 200 GB
Volume mount:     /workspace
Container:        runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

The training code is GPU-SKU neutral. Prefer the least-cost available 48-GB card; move to A100/H100 only after measured throughput shows the cheaper class is the bottleneck.

## Environment

Set these on the Pod. Keep `HF_TOKEN` in the RunPod secret/environment mechanism rather than the repository or startup command history.

```text
RUNPOD_PERSIST_ROOT=/workspace
VOICE_EXECUTION_ROOT=/workspace/sovereign-voice
EXEC_LANGUAGE=tw
MIN_FREE_GB=150
HF_TOKEN=<optional gated Hugging Face access token>
```

Optional controls:

```text
REQUIRE_EXTERNAL_EVAL=1
FORCE_REACQUIRE=1
REFRESH_SOURCE_LOCKS=1
VOICE_SOURCE_REF=main
```

`REQUIRE_EXTERNAL_EVAL=1` should only be enabled when the language's independent evaluation source is actually accessible. A missing gated benchmark must not prevent the public training corpus from being frozen and inspected unless that strict gate is intentional.

## Startup

From a fresh Pod shell:

```bash
git clone https://github.com/AbrahamBaffoe1/sovereign-voice-platform.git /workspace/bootstrap-src
cd /workspace/bootstrap-src
bash scripts/runpod_bootstrap.sh
```

The bootstrap creates/reuses a persistent source checkout and then hands execution to `scripts/run_real_training.sh`.

The launcher applies `constraints/training-cu124.txt` by default. That policy pins PyTorch 2.6 to
CUDA 12.4, verifies that `torch.cuda` sees the assigned GPU, and stores both the resolved environment
and the applied constraint under `state/`. A different reviewed stack must provide both a local
`PIP_CONSTRAINT` file and the matching `EXPECTED_TORCH_CUDA` value.

The child launcher performs the real machine preflight before corpus/model work:

```text
NVIDIA runtime visible
  -> Python 3.11 available
  -> persistent execution root created
  -> isolated persistent venv
  -> project training dependencies installed
  -> resolved dependency snapshot recorded
  -> corpus pipeline
  -> TTS readiness evidence
  -> ASR training/resume
  -> CTranslate2 export
  -> evaluation
```

## Durable outputs

Everything that matters survives Pod replacement under:

```text
/workspace/sovereign-voice/
  data/bootstrap/
  artifacts/bootstrap/
  artifacts/experiments/asr/
  artifacts/tts-readiness/
  cache/
  logs/real-execution.log
  state/EXECUTION_ENVIRONMENT.json
  state/REAL_EXECUTION.json
  state/pip-freeze.txt
  state/pip-constraint.txt
  state/execution.lock
  venv/
```

Do not move trained checkpoints to the container's ephemeral filesystem. Do not use GitHub Actions artifacts as the canonical corpus/model store.

## First-run acceptance checks

Before launching another language, review the Twi run for:

1. corpus-v0 has accepted rows and a stable fingerprint;
2. rejected-row reasons are explainable rather than dominated by an implementation bug;
3. normalized audio satisfies the 16-kHz mono ASR contract;
4. training and held-out audio have zero exact-hash leakage;
5. GPU memory has safe headroom at the chosen batch/accumulation settings;
6. interrupted-run resume was exercised or checkpoint structure was inspected;
7. final HF model and processor exist;
8. CTranslate2 export exists and can be loaded;
9. WER/CER reports exist for every available evaluation split;
10. recorded source/profile/corpus hashes match the intended release inputs.

TTS training remains linguistically gated. A successful corpus freeze may produce a candidate grapheme inventory, but no observed character automatically becomes an approved pronunciation rule.

## Cost guard

Stop or terminate the Pod when training/evaluation is not actively using the GPU. The persistent volume is what must remain; the compute instance is replaceable. Re-running the bootstrap on a new Pod should reuse the corpus, caches, venv, and compatible checkpoints instead of starting from zero.
