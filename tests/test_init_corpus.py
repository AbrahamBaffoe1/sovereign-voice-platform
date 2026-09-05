"""Tests for policy-bound corpus workspace initialization."""

import csv
from pathlib import Path

from training.init_corpus import BASE_COLUMNS, initialize_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_initialize_corpus_creates_strict_metadata_header(tmp_path: Path) -> None:
    """New target-language workspaces must start with governance columns, not ad-hoc CSVs."""
    output = tmp_path / "tw"
    initialize_corpus(ROOT / "training/configs/languages/tw.yaml", output)
    with (output / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == BASE_COLUMNS
    assert (output / "CORPUS_POLICY.json").exists()
    assert (output / "wavs/.gitkeep").exists()
