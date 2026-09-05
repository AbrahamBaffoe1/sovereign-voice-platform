"""Tests for persistent execution safety without requiring a GPU or network access."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.execution.environment import WorkspaceLayout, assert_persistent_workspace, preflight


def test_workspace_caches_are_pinned_under_durable_root(tmp_path: Path) -> None:
    """Provider/model caches must follow the execution disk rather than a disposable runner HOME."""
    layout = WorkspaceLayout.from_root(tmp_path / "voice-workspace")
    layout.create()
    env = layout.child_environment()

    assert Path(env["HF_HOME"]).is_relative_to(layout.cache_root)
    assert Path(env["HF_DATASETS_CACHE"]).is_relative_to(layout.cache_root)
    assert Path(env["TORCH_HOME"]).is_relative_to(layout.cache_root)
    assert Path(env["XDG_CACHE_HOME"]).is_relative_to(layout.cache_root)


def test_github_checkout_is_rejected_as_real_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkout cleanup must never be able to erase a long-running corpus or checkpoint tree."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))

    with pytest.raises(RuntimeError, match="disposable path"):
        assert_persistent_workspace(checkout / "training-data")


def test_preflight_can_use_ephemeral_storage_only_when_explicit(tmp_path: Path) -> None:
    """Smoke tests may opt into temporary storage, but the normal production path stays fail-closed."""
    layout = WorkspaceLayout.from_root(tmp_path / "smoke")
    report = preflight(layout, min_free_gb=0, require_gpu=False, allow_ephemeral=True)

    assert report["workspace"]["root"] == str(layout.root)
    assert Path(layout.state_root / "EXECUTION_ENVIRONMENT.json").exists()
