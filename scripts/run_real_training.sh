#!/usr/bin/env bash
set -Eeuo pipefail

# This launcher is intentionally provider-neutral. Run it inside a GPU VM/pod instead of registering
# that machine as a GitHub runner when the repository visibility or runner trust boundary is unsuitable.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_EXECUTION_ROOT="${VOICE_EXECUTION_ROOT:-/workspace/sovereign-voice}"
EXEC_LANGUAGE="${EXEC_LANGUAGE:-all}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

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
mkdir -p "$VOICE_EXECUTION_ROOT/state"
if [[ ! -x "$VOICE_EXECUTION_ROOT/venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VOICE_EXECUTION_ROOT/venv"
fi
source "$VOICE_EXECUTION_ROOT/venv/bin/activate"

cd "$REPO_ROOT"
python -m pip install --upgrade pip wheel
pip install -e '.[data,training-asr,asr,training]'

# Persist the exact Python environment next to the model lineage. This is diagnostic evidence for a
# future reproduction/debugging pass; pyproject.toml remains the dependency policy, not this snapshot.
python -m pip freeze > "$VOICE_EXECUTION_ROOT/state/pip-freeze.txt"
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
