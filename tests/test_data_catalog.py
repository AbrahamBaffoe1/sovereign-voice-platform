"""License-boundary tests for external training sources."""

from pathlib import Path

from training.data.catalog import SourceCatalog

ROOT = Path(__file__).resolve().parents[1]


def test_research_only_source_cannot_enter_production() -> None:
    """A research source must never be returned by the production data plan."""
    catalog = SourceCatalog(ROOT / "training/configs/source_catalog.yaml")
    production = {source.source_id for source in catalog.plan(language="tw", usage="production")}
    assert "kasa42_research" not in production
    assert "first_party_team" in production
