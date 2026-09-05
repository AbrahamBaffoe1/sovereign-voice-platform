#!/usr/bin/env bash
set -Eeuo pipefail

# This script provisions infrastructure only. The Mac remains the control plane; all CUDA training
# still happens inside the RunPod pod. Keeping provisioning separate from training makes it safe to
# recreate compute without changing corpus/model lineage.

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY in your shell; never place it in this repository}"
: "${RUNPOD_DC_ID:?Set RUNPOD_DC_ID from: runpodctl datacenter list}"
: "${RUNPOD_GPU_ID:?Set RUNPOD_GPU_ID from: runpodctl gpu list}"

RUNPOD_VOLUME_SIZE_GB="${RUNPOD_VOLUME_SIZE_GB:-200}"
RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-sovereign-voice-twi}"
RUNPOD_VOLUME_NAME="${RUNPOD_VOLUME_NAME:-sovereign-voice-data}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
RUNPOD_CONTAINER_DISK_GB="${RUNPOD_CONTAINER_DISK_GB:-30}"

if ! command -v runpodctl >/dev/null 2>&1; then
  cat >&2 <<'EOF'
runpodctl is not installed.
On macOS:
  brew install runpod/runpodctl/runpodctl
Then run:
  runpodctl update
  runpodctl version
EOF
  exit 2
fi

# RunPod returns JSON for resource commands. Parse resource IDs instead of scraping human-readable
# output so the script remains safe if the CLI adds fields or changes formatting.
json_id() {
  python3 -c '
import json, sys
obj = json.load(sys.stdin)
for key in ("id", "podId", "pod_id", "networkVolumeId", "network_volume_id"):
    value = obj.get(key) if isinstance(obj, dict) else None
    if value:
        print(value)
        raise SystemExit(0)
raise SystemExit("resource response did not contain a recognizable id")
'
}

printf 'Creating %s GiB persistent network volume in %s...\n' "$RUNPOD_VOLUME_SIZE_GB" "$RUNPOD_DC_ID"
volume_json="$(runpodctl network-volume create \
  --name "$RUNPOD_VOLUME_NAME" \
  --size "$RUNPOD_VOLUME_SIZE_GB" \
  --data-center-id "$RUNPOD_DC_ID")"
volume_id="$(printf '%s' "$volume_json" | json_id)"
printf 'Created network volume: %s\n' "$volume_id"

# The volume and pod are pinned to the same data center because RunPod network volumes are
# location-bound. /workspace is the durable boundary expected by scripts/runpod_bootstrap.sh.
printf 'Creating GPU pod on %s with %s...\n' "$RUNPOD_DC_ID" "$RUNPOD_GPU_ID"
pod_json="$(runpodctl pod create \
  --name "$RUNPOD_POD_NAME" \
  --gpu-id "$RUNPOD_GPU_ID" \
  --image "$RUNPOD_IMAGE" \
  --container-disk-in-gb "$RUNPOD_CONTAINER_DISK_GB" \
  --data-center-ids "$RUNPOD_DC_ID" \
  --network-volume-id "$volume_id" \
  --volume-mount-path /workspace \
  --ssh)"
pod_id="$(printf '%s' "$pod_json" | json_id)"
printf 'Created pod: %s\n' "$pod_id"

cat <<EOF

Infrastructure is provisioned.

POD_ID=$pod_id
VOLUME_ID=$volume_id

Next, wait until the pod runtime is ready:
  runpodctl pod get $pod_id

Then inspect SSH connection details:
  runpodctl ssh info $pod_id

Inside the pod, launch the Twi production lane:
  git clone https://github.com/AbrahamBaffoe1/sovereign-voice-platform.git /workspace/bootstrap-src
  cd /workspace/bootstrap-src
  export EXEC_LANGUAGE=tw
  export VOICE_EXECUTION_ROOT=/workspace/sovereign-voice
  export MIN_FREE_GB=150
  bash scripts/runpod_bootstrap.sh

Cost guard when you are done:
  runpodctl pod stop $pod_id

Stopping the pod stops GPU billing, but the persistent network volume remains so checkpoints and
corpora survive. Delete the pod only when you no longer need that compute definition; delete the
network volume only after you have intentionally archived or discarded the training state.
EOF
