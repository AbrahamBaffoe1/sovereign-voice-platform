"""Atomic model-promotion and rollback tests."""

from pathlib import Path

from training.deploy.model_registry import ModelRegistry


def test_model_registry_promotes_and_rolls_back(tmp_path: Path) -> None:
    """Production rollback should change only a pointer, never either immutable artifact."""
    registry = ModelRegistry(tmp_path / "models")
    artifact_a = tmp_path / "a.bin"
    artifact_b = tmp_path / "b.bin"
    artifact_a.write_bytes(b"a")
    artifact_b.write_bytes(b"b")
    registry.register(task="asr", language="tw", model_id="a", artifact_path=artifact_a, metadata={})
    registry.register(task="asr", language="tw", model_id="b", artifact_path=artifact_b, metadata={})
    registry.promote(task="asr", language="tw", model_id="a", stage="production")
    current = registry.promote(task="asr", language="tw", model_id="b", stage="production")
    assert current.previous_model_id == "a"
    rolled = registry.rollback(task="asr", language="tw")
    assert rolled.model_id == "a"
