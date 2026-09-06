"""Regression tests for the production RunPod dependency contract."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _dependency_for(extra: str, package: str) -> str:
    """Return one package requirement from a named optional-dependency group."""
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = project["optional-dependencies"][extra]
    return next(requirement for requirement in requirements if requirement.startswith(package))


def test_training_extras_cannot_float_to_a_new_torch_cuda_major() -> None:
    """Every production training extra must retain the reviewed torch upper bound."""
    assert _dependency_for("training-asr", "torch") == "torch>=2.6,<2.7"
    assert _dependency_for("tts-chatterbox", "torch") == "torch>=2.6,<2.7"
    assert _dependency_for("tts-chatterbox", "torchaudio") == "torchaudio>=2.6,<2.7"
    assert _dependency_for("tts-nemo", "torch") == "torch>=2.6,<2.7"
    assert _dependency_for("tts-voxcpm", "torch") == "torch>=2.5,<2.7"


def test_runpod_constraint_keeps_torch_and_torchaudio_aligned() -> None:
    """The concrete RunPod lock must use matching torch and torchaudio releases."""
    constraint = (REPOSITORY_ROOT / "constraints" / "training-cu124.txt").read_text(encoding="utf-8")

    assert "torch==2.6.0" in constraint.splitlines()
    assert "torchaudio==2.6.0" in constraint.splitlines()


def test_real_training_launcher_applies_constraint_before_install() -> None:
    """The production launcher must export its constraint before resolving dependencies."""
    launcher = (REPOSITORY_ROOT / "scripts" / "run_real_training.sh").read_text(encoding="utf-8")

    export_position = launcher.index("export PIP_CONSTRAINT")
    install_position = launcher.index("python -m pip install -e")
    assert export_position < install_position
    assert "constraints/training-cu124.txt" in launcher
    assert 'EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.4}"' in launcher


def test_runpod_provisioner_defaults_to_verified_cuda_124_image() -> None:
    """The A40 provisioning default must match the CUDA version enforced by the launcher."""
    provisioner = (REPOSITORY_ROOT / "scripts" / "mac_provision_runpod.sh").read_text(encoding="utf-8")

    assert "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" in provisioner


def test_self_hosted_workflow_uses_the_same_dependency_contract() -> None:
    """The GitHub runner fallback must not bypass the RunPod dependency policy."""
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "real-execution.yml").read_text(encoding="utf-8")

    assert "PIP_CONSTRAINT: ${{ github.workspace }}/constraints/training-cu124.txt" in workflow
    assert 'EXPECTED_TORCH_CUDA: "12.4"' in workflow
    assert 'cp "$PIP_CONSTRAINT" "$VOICE_EXECUTION_ROOT/state/pip-constraint.txt"' in workflow
