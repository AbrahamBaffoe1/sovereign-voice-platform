#!/usr/bin/env bash
set -Eeuo pipefail

# RunPod pods are disposable compute; the mounted /workspace volume is the durable boundary.
# Keeping the repository checkout under that mount makes source provenance and model state survive
# pod replacement without treating the container filesystem as authoritative storage.
PERSIST_ROOT="${RUNPOD_PERSIST_ROOT:-/workspace}"
CHECKOUT_DIR="${RUNPOD_CHECKOUT_DIR:-$PERSIST_ROOT/sovereign-voice-platform-src}"
EXECUTION_ROOT="${VOICE_EXECUTION_ROOT:-$PERSIST_ROOT/sovereign-voice}"
SOURCE_REF="${VOICE_SOURCE_REF:-main}"
REPOSITORY_URL="${VOICE_REPOSITORY_URL:-https://github.com/AbrahamBaffoe1/sovereign-voice-platform.git}"

if [[ ! -d "$PERSIST_ROOT" ]]; then
  echo "Persistent RunPod mount is missing: $PERSIST_ROOT" >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required in the RunPod image" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA runtime detected; refusing to start an expensive training bootstrap on CPU" >&2
  exit 2
fi

# HF_TOKEN is intentionally consumed only from the environment. Never place gated-dataset tokens
# in this repository, command history, or generated lineage files.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; public corpora can proceed, but gated evaluation sources may be skipped" >&2
fi

if [[ ! -d "$CHECKOUT_DIR/.git" ]]; then
  git clone "$REPOSITORY_URL" "$CHECKOUT_DIR"
fi

cd "$CHECKOUT_DIR"
git fetch --prune origin

# A symbolic ref such as main is resolved to one concrete commit before training. The child launcher
# records the resulting SHA, so every checkpoint can be traced back to the exact code that produced it.
git checkout --detach "origin/$SOURCE_REF" 2>/dev/null || git checkout --detach "$SOURCE_REF"
RESOLVED_SHA="$(git rev-parse HEAD)"
echo "Resolved training source: $RESOLVED_SHA"

export VOICE_EXECUTION_ROOT="$EXECUTION_ROOT"
export EXEC_LANGUAGE="${EXEC_LANGUAGE:-tw}"
export MIN_FREE_GB="${MIN_FREE_GB:-150}"

# exec preserves signals from RunPod's process supervisor. The Python orchestrator owns checkpoint
# resume and durable phase state; this wrapper must not hide termination behind a background shell.
exec bash scripts/run_real_training.sh
