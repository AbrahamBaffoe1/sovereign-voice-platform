#!/usr/bin/env bash
set -Eeuo pipefail

# This launcher is intentionally provider-neutral. Run it inside a GPU VM/pod instead of registering
# that machine as a GitHub runner when the repository visibility or runner trust boundary is unsuitable.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_EXECUTION_ROOT="${VOICE_EXECUTION_ROOT:-/workspace/sovereign-voice}"
EXEC_LANGUAGE="${EXEC_LANGUAGE:-all}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.4}"

# Production installs must not float to a newly published PyTorch/CUDA major version. Operators may
# provide a reviewed local constraint file, but the verified RunPod lane defaults to torch 2.6 + cu124.
if [[ -z "${PIP_CONSTRAINT:-}" ]]; then
  PIP_CONSTRAINT="$REPO_ROOT/constraints/training-cu124.txt"
fi
if [[ ! -f "$PIP_CONSTRAINT" ]]; then
  echo "Python dependency constraint does not exist: $PIP_CONSTRAINT" >&2
  exit 2
fi
export PIP_CONSTRAINT

# Dependency downloads are expensive and safe to reuse. Keep the pip cache beside the durable corpus
# and checkpoints rather than in the disposable container filesystem.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$VOICE_EXECUTION_ROOT/cache/pip}"
export PIP_CACHE_DIR

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for real ASR training" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required; set PYTHON_BIN to the Python 3.11 executable" >&2
  exit 2
fi

# The virtual environment, package caches, corpus and checkpoints all live on the mounted execution
# volume. Recreating a cloud pod therefore does not force the project to download or train from zero.
mkdir -p "$VOICE_EXECUTION_ROOT/state" "$PIP_CACHE_DIR"
if [[ ! -x "$VOICE_EXECUTION_ROOT/venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VOICE_EXECUTION_ROOT/venv"
fi
source "$VOICE_EXECUTION_ROOT/venv/bin/activate"

cd "$REPO_ROOT"
echo "Dependency constraint: $PIP_CONSTRAINT"
python -m pip install --upgrade pip wheel
python -m pip install -e '.[data,training-asr,asr,training]'
python -m pip check

# Fail before corpus acquisition if the resolved wheel targets a different CUDA runtime or cannot see
# the assigned GPU. This catches image/dependency drift while the run is still cheap to restart.
EXPECTED_TORCH_CUDA="$EXPECTED_TORCH_CUDA" python - <<'PY'
import os

import torch

expected_cuda = os.environ["EXPECTED_TORCH_CUDA"]
resolved_cuda = torch.version.cuda or ""
if resolved_cuda != expected_cuda:
    raise SystemExit(f"Expected torch CUDA {expected_cuda}, resolved {resolved_cuda or 'none'}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access the assigned NVIDIA GPU")
print(
    "Verified training runtime: "
    f"torch={torch.__version__} cuda={resolved_cuda} gpu={torch.cuda.get_device_name(0)}"
)
PY

# Persist the exact Python environment next to the model lineage. This is diagnostic evidence for a
# future reproduction/debugging pass; pyproject.toml remains the dependency policy, not this snapshot.
python -m pip freeze > "$VOICE_EXECUTION_ROOT/state/pip-freeze.txt"
cp "$PIP_CONSTRAINT" "$VOICE_EXECUTION_ROOT/state/pip-constraint.txt"
export VOICE_SOURCE_SHA="$(git rev-parse HEAD 2>/dev/null || true)"

args=(
  --workspace "$VOICE_EXECUTION_ROOT"
  --language "$EXEC_LANGUAGE"
  --min-free-gb "$MIN_FREE_GB"
)
if [[ "${REQUIRE_EXTERNAL_EVAL:-0}" == "1" ]]; then
  args+=(--require-external-eval)
fi
if [[ "${FORCE_REACQUIRE:-0}" == "1" ]]; then
  args+=(--force-reacquire)
fi
if [[ "${REFRESH_SOURCE_LOCKS:-0}" == "1" ]]; then
  args+=(--refresh-source-locks)
fi

# Do not wrap this in nohup/backgrounding here. The provider/process supervisor should own process
# lifetime, while the Python orchestrator owns resumability and durable phase state.
python -m training.execution.run_pipeline "${args[@]}"
