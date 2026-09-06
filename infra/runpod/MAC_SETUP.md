# Provision RunPod from a MacBook

Your Apple Silicon Mac is the **control plane**, not the CUDA training machine. The Mac installs `runpodctl`, creates the RunPod volume/pod, and opens SSH. The NVIDIA GPU inside RunPod performs corpus processing that requires CUDA, Whisper training, CTranslate2 export, and later TTS training.

## 1. Install the current RunPod CLI

```bash
brew install runpod/runpodctl/runpodctl
runpodctl update
runpodctl version
```

Live CLI help is authoritative for flags:

```bash
runpodctl pod create --help
runpodctl network-volume create --help
```

## 2. Create an API key locally

Create a RunPod API key in the RunPod console and keep it on your Mac. Do **not** paste the key into this repository, GitHub issues, model metadata, or chat transcripts.

For one terminal session:

```bash
export RUNPOD_API_KEY='...'
```

For an interactive first-time setup including SSH keys, RunPod also provides:

```bash
runpodctl doctor
```

Verify authentication before creating paid resources:

```bash
runpodctl pod list
```

## 3. Choose one data center and one 48 GB GPU

```bash
runpodctl datacenter list
runpodctl gpu list
```

For the first Whisper-small baseline prefer the least-cost available 48 GB card in one data center, for example A40 or RTX A6000. The exact SKU is intentionally not hard-coded because stock and pricing change.

Set the values returned by the CLI:

```bash
export RUNPOD_DC_ID='US-...'
export RUNPOD_GPU_ID='NVIDIA A40'
```

The GPU and network volume must be in the same data center.

## 4. Provision the persistent volume and GPU pod

From a checkout of this repository:

```bash
bash scripts/mac_provision_runpod.sh
```

Defaults:

```text
network volume:     200 GiB
volume mount:       /workspace
container disk:     30 GiB
pod name:           sovereign-voice-twi
execution language: Twi first
container image:    PyTorch 2.4 / Python 3.11 / CUDA 12.4.1
```

Optional overrides:

```bash
export RUNPOD_VOLUME_SIZE_GB=300
export RUNPOD_POD_NAME=sovereign-voice-twi-v1
export RUNPOD_VOLUME_NAME=sovereign-voice-data-v1
export RUNPOD_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

The script prints the resulting `POD_ID` and `VOLUME_ID` and does not write your API key to disk.

## 5. Connect to the pod

Wait for the runtime:

```bash
runpodctl pod get <POD_ID>
```

Get SSH details:

```bash
runpodctl ssh info <POD_ID>
```

Use the returned SSH command/key to connect from the Mac.

## 6. Start the Twi execution lane inside RunPod

```bash
git clone https://github.com/AbrahamBaffoe1/sovereign-voice-platform.git /workspace/bootstrap-src
cd /workspace/bootstrap-src

export EXEC_LANGUAGE=tw
export VOICE_EXECUTION_ROOT=/workspace/sovereign-voice
export MIN_FREE_GB=150

# Optional. Needed only for gated Hugging Face sources that your account has accepted.
export HF_TOKEN='...'

bash scripts/runpod_bootstrap.sh
```

The pipeline then owns acquisition, corpus-v0 freezing, leakage checks, TTS readiness evidence, Whisper-small training/resume, CTranslate2 export, and ASR evaluation.

The bootstrap defaults to the reviewed `constraints/training-cu124.txt` dependency policy and refuses
to start corpus work unless PyTorch resolves CUDA 12.4 and can access the assigned NVIDIA GPU.

## 7. Stop compute without losing training state

When the job is not running:

```bash
runpodctl pod stop <POD_ID>
```

The network volume remains and retains corpora, caches, checkpoints, logs, and lineage. Stop/delete semantics are intentionally separated from volume deletion so shutting down GPU billing cannot erase model state.

Before deleting the network volume, intentionally archive or discard the contents. Network-volume deletion is destructive.
