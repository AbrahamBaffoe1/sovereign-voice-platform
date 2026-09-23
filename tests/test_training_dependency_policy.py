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


def _project_dependency(package: str) -> str:
    """Return one base project dependency."""
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return next(requirement for requirement in project["dependencies"] if requirement.startswith(package))


def test_base_numpy_policy_remains_chatterbox_compatible() -> None:
    """The shared package metadata must not force Chatterbox into NumPy 2.x."""
    assert _project_dependency("numpy") == "numpy>=1.26,<3"


def test_training_extras_cannot_float_to_a_new_torch_cuda_major() -> None:
    """Every production training extra must retain the reviewed torch upper bound."""
    assert _dependency_for("training-asr", "torch") == "torch>=2.6,<2.7"
    assert _dependency_for("tts-chatterbox", "torch") == "torch>=2.6,<2.7"
    assert _dependency_for("tts-chatterbox", "torchaudio") == "torchaudio>=2.6,<2.7"
    assert _dependency_for("tts-nemo", "torch") == "torch>=2.6,<2.7"
    assert _dependency_for("tts-voxcpm", "torch") == "torch>=2.5,<2.7"


def test_training_asr_extra_stays_on_transformers_4() -> None:
    """The validated ASR lane must not resolve into the Chatterbox Transformers stack."""
    assert _dependency_for("training-asr", "transformers") == "transformers>=4.55,<5"
    assert _dependency_for("tts-chatterbox", "chatterbox-tts") == "chatterbox-tts>=0.1.7,<0.2"


def test_runpod_training_constraint_keeps_asr_stack_aligned() -> None:
    """The concrete RunPod ASR lock must use the reviewed CUDA and Transformers stack."""
    constraint = (REPOSITORY_ROOT / "constraints" / "training-cu124.txt").read_text(encoding="utf-8")

    assert "torch==2.6.0" in constraint.splitlines()
    assert "torchaudio==2.6.0" in constraint.splitlines()
    assert "transformers==4.57.6" in constraint.splitlines()
    assert "accelerate==1.14.0" in constraint.splitlines()


def test_chatterbox_constraint_is_separate_from_asr_training() -> None:
    """Chatterbox must retain its own CUDA 12.4 environment contract."""
    constraint = (REPOSITORY_ROOT / "constraints" / "chatterbox-cu124.txt").read_text(encoding="utf-8")

    assert "chatterbox-tts==0.1.7" in constraint.splitlines()
    assert "numpy==1.26.4" in constraint.splitlines()
    assert "torch==2.6.0" in constraint.splitlines()
    assert "torchaudio==2.6.0" in constraint.splitlines()
    assert "transformers==5.2.0" in constraint.splitlines()


def test_makefile_keeps_asr_and_chatterbox_installs_isolated() -> None:
    """Convenience install targets must not combine incompatible extras."""
    makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "install-asr:" in makefile
    assert "install-tts-chatterbox:" in makefile
    assert "constraints/training-cu124.txt" in makefile
    assert "constraints/chatterbox-cu124.txt" in makefile
    assert "install-all:" not in makefile
    assert "asr,tts-chatterbox" not in makefile


def test_uv_metadata_declares_incompatible_training_extra_conflicts() -> None:
    """uv resolves all extras unless these conflicts are explicit."""
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    conflicts = project["tool"]["uv"]["conflicts"]

    assert [
        {"extra": "training-asr"},
        {"extra": "tts-chatterbox"},
    ] in conflicts
    assert [
        {"extra": "data"},
        {"extra": "tts-chatterbox"},
    ] in conflicts
    assert [
        {"extra": "data"},
        {"extra": "tts-voxcpm"},
    ] in conflicts
    assert [
        {"extra": "training-asr"},
        {"extra": "tts-voxcpm"},
    ] in conflicts


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
