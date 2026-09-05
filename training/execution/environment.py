"""Machine/workspace preflight for expensive training jobs.

The checks here are intentionally standard-library first. They run before network downloads or GPU
allocation so a misconfigured runner fails cheaply instead of leaving a half-built corpus behind.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    """Stable directory contract shared by corpus acquisition, experiments, caches and run state."""

    root: Path
    data_root: Path
    artifacts_root: Path
    experiments_root: Path
    tts_readiness_root: Path
    state_root: Path
    logs_root: Path
    cache_root: Path

    @classmethod
    def from_root(cls, root: Path) -> WorkspaceLayout:
        """Resolve one operator-owned root into paths that survive checkout cleanup and process restarts."""
        resolved = root.expanduser().resolve()
        return cls(
            root=resolved,
            data_root=resolved / "data" / "bootstrap",
            artifacts_root=resolved / "artifacts" / "bootstrap",
            experiments_root=resolved / "artifacts" / "experiments" / "asr",
            tts_readiness_root=resolved / "artifacts" / "tts-readiness",
            state_root=resolved / "state",
            logs_root=resolved / "logs",
            cache_root=resolved / "cache",
        )

    def create(self) -> None:
        """Create all durable roots before a phase starts so later failures still have somewhere to report."""
        for path in (
            self.root,
            self.data_root,
            self.artifacts_root,
            self.experiments_root,
            self.tts_readiness_root,
            self.state_root,
            self.logs_root,
            self.cache_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def child_environment(self) -> dict[str, str]:
        """Pin provider/model caches under the durable workspace instead of a runner's disposable HOME."""
        cache = self.cache_root
        env = os.environ.copy()
        env.update(
            {
                "HF_HOME": str(cache / "huggingface"),
                "HF_DATASETS_CACHE": str(cache / "huggingface" / "datasets"),
                "TORCH_HOME": str(cache / "torch"),
                "XDG_CACHE_HOME": str(cache / "xdg"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env


def _is_within(path: Path, parent: Path) -> bool:
    """Return true when path is parent itself or any descendant, without relying on string prefixes."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_persistent_workspace(root: Path, *, allow_ephemeral: bool = False) -> None:
    """Reject well-known disposable locations that would erase downloaded corpora/checkpoints after a job."""
    if allow_ephemeral:
        return
    resolved = root.expanduser().resolve()
    disposable: list[Path] = [Path("/tmp"), Path("/var/tmp")]
    for variable in ("RUNNER_TEMP", "GITHUB_WORKSPACE"):
        value = os.environ.get(variable)
        if value:
            disposable.append(Path(value).expanduser().resolve())
    for candidate in disposable:
        if _is_within(resolved, candidate):
            raise RuntimeError(
                f"execution workspace {resolved} is inside disposable path {candidate}; "
                "use a mounted persistent disk or pass --allow-ephemeral only for smoke tests"
            )


def disk_report(root: Path) -> dict[str, float | str]:
    """Measure the filesystem that actually backs the execution root after it has been created."""
    usage = shutil.disk_usage(root)
    return {
        "path": str(root),
        "total_gb": round(usage.total / _GIB, 3),
        "used_gb": round(usage.used / _GIB, 3),
        "free_gb": round(usage.free / _GIB, 3),
    }


def gpu_report(*, required: bool) -> dict[str, object]:
    """Verify both the NVIDIA driver boundary and PyTorch CUDA visibility before ASR training starts."""
    command = shutil.which("nvidia-smi")
    if command is None:
        if required:
            raise RuntimeError("nvidia-smi is unavailable; real ASR training requires an NVIDIA GPU runner")
        return {"required": required, "available": False, "reason": "nvidia-smi not found"}

    completed = subprocess.run(
        [
            command,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if required and not devices:
        raise RuntimeError("nvidia-smi returned no GPUs")

    torch_details: dict[str, object] = {"installed": False, "cuda_available": False}
    try:
        import torch
    except ImportError:
        if required:
            raise RuntimeError("PyTorch is not installed in the execution environment") from None
    else:
        cuda_available = bool(torch.cuda.is_available())
        torch_details = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ] if cuda_available else [],
        }
        if required and not cuda_available:
            raise RuntimeError("NVIDIA driver is visible but torch.cuda.is_available() is false")

    return {
        "required": required,
        "available": bool(devices),
        "nvidia_smi": devices,
        "torch": torch_details,
    }


def preflight(
    layout: WorkspaceLayout,
    *,
    min_free_gb: float,
    require_gpu: bool,
    allow_ephemeral: bool = False,
) -> dict[str, object]:
    """Create and validate the durable execution boundary, returning a report suitable for audit logs."""
    if min_free_gb < 0:
        raise ValueError("min_free_gb cannot be negative")
    assert_persistent_workspace(layout.root, allow_ephemeral=allow_ephemeral)
    layout.create()
    usage = disk_report(layout.root)
    if float(usage["free_gb"]) < min_free_gb:
        raise RuntimeError(
            f"workspace has {usage['free_gb']} GiB free; execution requires at least {min_free_gb} GiB"
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "workspace": {key: str(value) for key, value in asdict(layout).items()},
        "disk": usage,
        "gpu": gpu_report(required=require_gpu),
    }
    path = layout.state_root / "EXECUTION_ENVIRONMENT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


@contextmanager
def execution_lock(layout: WorkspaceLayout) -> Iterator[Path]:
    """Hold an OS file lock so two expensive executions cannot mutate the same corpus/checkpoint tree."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - real training is currently Linux-only.
        raise RuntimeError("execution locking requires a POSIX host") from exc

    layout.state_root.mkdir(parents=True, exist_ok=True)
    lock_path = layout.state_root / "execution.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another execution already owns workspace lock: {lock_path}") from exc

        # The file content is only operator context; the kernel lock on the open descriptor is authoritative.
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )
        handle.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
