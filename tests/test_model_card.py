"""Tests for checkpoint hashing and immutable model identifiers."""

from pathlib import Path

from training.common.model_card import build_model_id, hash_checkpoint_tree


def test_checkpoint_hash_is_stable_and_ignores_model_card(tmp_path: Path) -> None:
    """Model identity should reflect model files, not metadata written after hashing."""
    (tmp_path / "weights.bin").write_bytes(b"abc")
    first, files = hash_checkpoint_tree(tmp_path)
    (tmp_path / "model_card.json").write_text("{}", encoding="utf-8")
    second, _ = hash_checkpoint_tree(tmp_path)
    assert first == second
    assert files[0]["path"] == "weights.bin"
    assert build_model_id("tw", "asr", first).startswith("tw-asr-")
