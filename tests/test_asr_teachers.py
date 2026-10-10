from pathlib import Path

import pytest

from training.asr.pseudo_label import _training_rows
from training.asr.teacher_benchmark import _apply_language_prefix, _manifest_rows, _normalized_error_summary
from training.asr.teacher_registry import get_teacher, load_teacher_registry
from training.benchmarks.speech import BenchmarkUtterance


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_teacher_registry_exposes_public_and_gated_candidates() -> None:
    teachers = load_teacher_registry()
    assert teachers["dondo-twi-v2"].runnable is True
    assert teachers["dondo-twi-v2"].language_prefix_id == 2
    assert teachers["tubastt-twi-en-v0.5"].family == "ctc"
    assert teachers["sunflower-51-african-languages"].runnable is False
    assert "approval" in (teachers["sunflower-51-african-languages"].blocked_reason or "").lower()


def test_unknown_teacher_fails_with_valid_names() -> None:
    with pytest.raises(KeyError, match="dondo-twi-v2"):
        get_teacher("missing")


def test_benchmark_selection_only_reads_requested_split(tmp_path: Path) -> None:
    manifest = tmp_path / "audit.jsonl"
    _write_jsonl(manifest, [
        '{"audio_filepath":"train.wav","text":"train","split":"train"}',
        '{"audio_path":"test-a.wav","text":"a","split":"test"}',
        '{"audio_filepath":"test-b.wav","text":"b","split":"test"}',
    ])
    rows = _manifest_rows(manifest, split="test", limit=1)
    assert [row["audio_filepath"] for row in rows] == ["test-a.wav"]


def test_pseudo_label_input_rejects_validation_or_test_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "mixed.jsonl"
    _write_jsonl(manifest, [
        '{"audio_filepath":"train.wav","text":"train","split":"train"}',
        '{"audio_filepath":"test.wav","text":"test","split":"test"}',
    ])
    with pytest.raises(ValueError, match="non-train split"):
        _training_rows(manifest, limit=None)


def test_pseudo_label_input_accepts_train_only_manifest_without_split_field(tmp_path: Path) -> None:
    manifest = tmp_path / "train.json"
    _write_jsonl(manifest, ['{"audio_filepath":"a.wav","text":"human"}'])
    rows = _training_rows(manifest, limit=None)
    assert len(rows) == 1


def test_dondo_language_prefix_prepends_one_hot_feature_frame() -> None:
    torch = pytest.importorskip("torch")
    inputs = {
        "input_features": torch.zeros((1, 3, 8)),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    prefixed = _apply_language_prefix(inputs, torch, 2)
    assert tuple(prefixed["input_features"].shape) == (1, 4, 8)
    assert prefixed["input_features"][0, 0, 2].item() == 1.0
    assert prefixed["input_features"][0, 0].sum().item() == 1.0
    assert tuple(prefixed["attention_mask"].shape) == (1, 4)


def test_compatibility_normalizer_changes_metric_without_changing_rows() -> None:
    rows = [BenchmarkUtterance(reference="Hello, WORLD!", hypothesis="hello world", audio_seconds=1.0, latency_seconds=0.1)]
    summary = _normalized_error_summary(rows, lambda text: text.replace(",", "").replace("!", "").casefold())
    assert summary["wer"] == 0.0
    assert summary["cer"] == 0.0
    assert summary["sentence_accuracy"] == 1.0


def test_pseudo_label_accepts_legacy_train_contract_and_audio_path(tmp_path: Path) -> None:
    manifest = tmp_path / "train-source.jsonl"
    _write_jsonl(
        manifest,
        [
            '{"audio_path":"a.flac","text":"human","source_split":"train",'
            '"operational_class":"TRAIN","allow_optimizer_updates":true}'
        ],
    )
    rows = _training_rows(manifest, limit=None)
    assert rows[0]["audio_filepath"] == "a.flac"


def test_pseudo_label_rejects_evaluation_only_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "eval.jsonl"
    _write_jsonl(
        manifest,
        [
            '{"audio_path":"a.mp3","text":"eval","split":"test",'
            '"operational_class":"EVALUATION_ONLY","allow_optimizer_updates":false}'
        ],
    )
    with pytest.raises(ValueError, match="non-train split"):
        _training_rows(manifest, limit=None)


def test_teacher_comparison_rejects_misaligned_predictions(tmp_path: Path) -> None:
    from training.asr.compare_teachers import compare_predictions

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_jsonl(left, ['{"audio_filepath":"a.wav","reference":"x","hypothesis":"x","teacher_confidence":0.9,"audio_seconds":1,"latency_seconds":0.1}'])
    _write_jsonl(right, ['{"audio_filepath":"b.wav","reference":"x","hypothesis":"x","teacher_confidence":0.9,"audio_seconds":1,"latency_seconds":0.1}'])
    with pytest.raises(ValueError, match="alignment mismatch"):
        compare_predictions(left, right)


def test_teacher_comparison_confidence_selector_uses_higher_confidence(tmp_path: Path) -> None:
    from training.asr.compare_teachers import compare_predictions

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_jsonl(left, ['{"audio_filepath":"a.wav","reference":"hello","hypothesis":"wrong","teacher_confidence":0.2,"audio_seconds":1,"latency_seconds":0.1}'])
    _write_jsonl(right, ['{"audio_filepath":"a.wav","reference":"hello","hypothesis":"hello","teacher_confidence":0.9,"audio_seconds":1,"latency_seconds":0.1}'])
    report = compare_predictions(left, right)
    assert report["confidence_selected"]["right_selected"] == 1
    assert report["confidence_selected"]["metrics"]["global"]["wer"] == 0.0
