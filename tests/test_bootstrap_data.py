"""Bootstrap-corpus policy tests that do not require network access or large speech downloads."""

from pathlib import Path

from training.data.acquire import dry_run_plan, resolve_revision
from training.data.bootstrap_plan import BootstrapPlan
from training.data.catalog import DataSource, SourceCatalog
from training.data.hf_parquet import required_parquet_columns

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_plan_separates_training_and_evaluation() -> None:
    """Reserved benchmark sources must never silently enter the training source list."""
    plan = BootstrapPlan(ROOT / "training/configs/bootstrap_corpora.yaml")
    assert plan.sources(language="tw", task="asr", role="train") == (
        "waxal_akan_asr",
        "twi_words_400k",
    )
    assert plan.sources(language="tw", task="asr", role="eval") == ("waxal_akan_eval",)
    assert "cdli_ga_standard_eval" not in plan.sources(language="gaa", task="asr", role="train")


def test_dry_run_exposes_pinned_revisions() -> None:
    """Operators should see exact immutable upstream revisions before a long acquisition starts."""
    catalog = SourceCatalog(ROOT / "training/configs/source_catalog.yaml")
    plan = BootstrapPlan(ROOT / "training/configs/bootstrap_corpora.yaml")
    rows = dry_run_plan(language="tw", task="asr", catalog=catalog, plan=plan, include_eval=True)
    by_source = {row["source_id"]: row for row in rows}
    assert by_source["waxal_akan_asr"]["revision"] == "5f4d8ca24f2b9d168b2ee545f1febaaff4b40580"
    assert by_source["twi_words_400k"]["revision"] == "e808e3c75f31d306b33f7adb79a22f5aa3ea28f1"
    assert by_source["waxal_akan_eval"]["role"] == "eval"


def test_production_plan_excludes_research_only_sources() -> None:
    """Useful research corpora remain ineligible for commercial/production training by default."""
    catalog = SourceCatalog(ROOT / "training/configs/source_catalog.yaml")
    production = {source.source_id for source in catalog.plan(language="tw", usage="production", task="asr")}
    assert "kasa42_research" not in production
    assert "waxal_akan_asr" in production


def test_noncommercial_ewe_navigation_is_excluded_from_production() -> None:
    """The current CC-BY-NC Ewe navigation corpus must stay out of the production bootstrap plan."""
    catalog = SourceCatalog(ROOT / "training/configs/source_catalog.yaml")
    plan = BootstrapPlan(ROOT / "training/configs/bootstrap_corpora.yaml")
    assert plan.sources(language="ee", task="asr", role="train") == ("waxal_ewe_asr",)
    assert plan.sources(language="ee", task="tts", role="train") == ("waxal_ewe_tts",)
    source = catalog.get("ewe_navigation_research")
    assert source.usage == "research"
    assert source.license == "CC-BY-NC-4.0"
    assert not source.governance_approved
    production = {item.source_id for item in catalog.plan(language="ee", usage="production", task="asr")}
    assert "ewe_navigation_research" not in production


def test_waxal_ewe_asr_uses_pinned_raw_parquet_route() -> None:
    """The upstream WAXAL Ewe index-column mismatch must not regress back to Dataset feature casting."""
    catalog = SourceCatalog(ROOT / "training/configs/source_catalog.yaml")
    train = catalog.get("waxal_ewe_asr")
    evaluation = catalog.get("waxal_ewe_eval")
    assert train.data_files_glob == "data/ASR/ewe/ewe-train-*.parquet"
    assert evaluation.data_files_glob == "data/ASR/ewe/ewe-test-*.parquet"
    assert required_parquet_columns(train) == (
        "audio",
        "transcription",
        "speaker_id",
        "id",
        "language",
    )
    assert "__index_level_0__" not in required_parquet_columns(train)


def test_existing_revision_lock_is_reused_without_network(tmp_path: Path) -> None:
    """A resolved source revision must stay frozen unless an operator explicitly refreshes the lock."""
    source = DataSource(
        source_id="example",
        provider="huggingface",
        repo_id="org/data",
        revision=None,
        config="default",
        split="train",
        languages=("tw",),
        tasks=("asr",),
        usage="production",
        license="CC-BY-4.0",
        requires_revision_pin=True,
    )
    lock = tmp_path / "locks/example.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        '{"repo_id":"org/data","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
        encoding="utf-8",
    )
    assert resolve_revision(source, output_root=tmp_path) == "a" * 40


def test_all_bootstrap_train_sources_have_governance_approval() -> None:
    """Every public source entering corpus-v0 must have an explicit governance decision in the catalog."""
    catalog = SourceCatalog(ROOT / "training/configs/source_catalog.yaml")
    plan = BootstrapPlan(ROOT / "training/configs/bootstrap_corpora.yaml")
    for language in ("tw", "gaa", "ee", "ha"):
        for task in ("asr", "tts"):
            for source_id in plan.sources(language=language, task=task, role="train"):
                source = catalog.get(source_id)
                assert source.governance_approved
                assert source.allows("production")
                assert source.supports(task)
