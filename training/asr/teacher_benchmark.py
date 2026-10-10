"""Benchmark public ASR teacher checkpoints on frozen manifests without altering the corpus."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from training.asr.teacher_registry import get_teacher
from training.benchmarks.speech import BenchmarkUtterance, benchmark_report, error_rate
from training.common.manifest import file_sha256, read_jsonl


def _load_audio(path: Path, *, sample_rate: int = 16000) -> tuple[np.ndarray, float]:
    """Decode one utterance, falling back to ffmpeg for formats unavailable in libsndfile."""
    try:
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if sr != sample_rate:
            raise ValueError(f"{path}: expected {sample_rate} Hz, found {sr}")
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise ValueError(f"{path}: invalid mono waveform")
        array = np.asarray(waveform, dtype=np.float32)
    except (RuntimeError, ValueError):
        process = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        array = np.frombuffer(process.stdout, dtype=np.float32).copy()
        if not len(array) or not np.isfinite(array).all():
            raise ValueError(f"{path}: ffmpeg produced invalid audio") from None
    return array, len(array) / float(sample_rate)


def _manifest_rows(path: Path, *, split: str | None, limit: int | None) -> list[dict[str, object]]:
    rows = read_jsonl(path)
    selected: list[dict[str, object]] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        audio_path = row.get("audio_filepath") or row.get("audio_path")
        if not audio_path or row.get("text") is None:
            raise ValueError(f"{path}: row missing audio path/text")
        normalized = dict(row)
        normalized["audio_filepath"] = str(audio_path)
        selected.append(normalized)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError(f"no benchmark rows selected from {path}")
    return selected



def _normalized_error_summary(rows: list[BenchmarkUtterance], normalizer: Any) -> dict[str, float | int]:
    """Aggregate WER/CER after an explicit text normalizer for compatibility comparisons."""
    word_errors = reference_words = char_errors = reference_chars = exact = 0
    for row in rows:
        reference = str(normalizer(row.reference))
        hypothesis = str(normalizer(row.hypothesis))
        edits, total = error_rate(reference, hypothesis, unit="word")
        word_errors += edits
        reference_words += total
        edits, total = error_rate(reference, hypothesis, unit="char")
        char_errors += edits
        reference_chars += total
        exact += int(reference == hypothesis)
    return {
        "utterances": len(rows),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "char_errors": char_errors,
        "reference_chars": reference_chars,
        "wer": word_errors / reference_words if reference_words else float("nan"),
        "cer": char_errors / reference_chars if reference_chars else float("nan"),
        "sentence_accuracy": exact / len(rows) if rows else float("nan"),
    }


def _whisper_basic_summary(rows: list[BenchmarkUtterance]) -> dict[str, float | int]:
    """Match the legacy Nexora/Whisper benchmark normalization for apples-to-apples gating."""
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    return _normalized_error_summary(rows, BasicTextNormalizer())

def _load_ctc_teacher(model_id: str, device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor
    except ImportError as exc:
        raise SystemExit("Install ASR training dependencies: pip install -e '.[training-asr]'") from exc
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id)
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(resolved_device)
    model.eval()
    return processor, model, torch


def _apply_language_prefix(tensor_inputs: dict[str, Any], torch: Any, lang_id: int | None) -> dict[str, Any]:
    """Prepend the DONDO one-hot acoustic language frame when a teacher declares one."""
    if lang_id is None:
        return tensor_inputs
    features = tensor_inputs.get("input_features")
    if features is None or features.ndim != 3:
        raise ValueError("language-prefixed teacher requires batched input_features")
    batch, frames, feature_dim = features.shape
    if lang_id >= feature_dim:
        raise ValueError(f"language prefix id {lang_id} exceeds feature dimension {feature_dim}")
    prefix = torch.zeros((batch, 1, feature_dim), dtype=features.dtype, device=features.device)
    prefix[:, 0, lang_id] = 1.0
    tensor_inputs["input_features"] = torch.cat([prefix, features], dim=1)
    mask = tensor_inputs.get("attention_mask")
    if mask is not None and mask.ndim == 2 and mask.shape[1] == frames:
        leading = torch.ones((batch, 1), dtype=mask.dtype, device=mask.device)
        tensor_inputs["attention_mask"] = torch.cat([leading, mask], dim=1)
    return tensor_inputs


def _ctc_transcribe(
    processor: Any,
    model: Any,
    torch: Any,
    waveform: np.ndarray,
    *,
    language_prefix_id: int | None = None,
) -> tuple[str, float]:
    """Decode one CTC utterance and return transcript plus mean frame posterior confidence."""
    resolved_device = next(model.parameters()).device
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    tensor_inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
    tensor_inputs = _apply_language_prefix(tensor_inputs, torch, language_prefix_id)
    with torch.inference_mode():
        logits = model(**tensor_inputs).logits
        probabilities = torch.softmax(logits.float(), dim=-1)
    frame_confidence, predicted = probabilities.max(dim=-1)
    transcript = str(processor.batch_decode(predicted)[0]).strip()
    confidence = float(frame_confidence.mean().detach().cpu().item())
    return transcript, confidence


def run_benchmark(
    *,
    teacher_name: str,
    manifest: Path,
    output_dir: Path,
    split: str | None = "test",
    limit: int | None = None,
    device: str = "auto",
    model_source: str | None = None,
) -> dict[str, object]:
    """Run one immutable teacher benchmark and persist row-level and aggregate evidence."""
    teacher = get_teacher(teacher_name)
    if not teacher.runnable:
        raise RuntimeError(teacher.blocked_reason or f"teacher {teacher_name} is not directly runnable")
    if teacher.family != "ctc":
        raise RuntimeError(f"teacher family {teacher.family!r} is not implemented yet")
    rows = _manifest_rows(manifest, split=split, limit=limit)
    resolved_model_source = model_source or teacher.model_id
    processor, model, torch = _load_ctc_teacher(resolved_model_source, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    benchmark_rows: list[BenchmarkUtterance] = []
    confidences: list[float] = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            audio_path = Path(str(row["audio_filepath"]))
            waveform, audio_seconds = _load_audio(audio_path)
            started = time.perf_counter()
            hypothesis, confidence = _ctc_transcribe(
                processor, model, torch, waveform, language_prefix_id=teacher.language_prefix_id
            )
            latency = time.perf_counter() - started
            utterance = BenchmarkUtterance(
                reference=str(row["text"]),
                hypothesis=hypothesis,
                audio_seconds=audio_seconds,
                latency_seconds=latency,
                speaker=str(row["speaker"]) if row.get("speaker") else None,
                dialect=str(row["dialect"]) if row.get("dialect") else None,
                noise=str(row["noise"]) if row.get("noise") else None,
                code_switch=str(row["code_switch"]) if row.get("code_switch") else None,
            )
            benchmark_rows.append(utterance)
            confidences.append(confidence)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "audio_filepath": str(audio_path),
                        "reference": utterance.reference,
                        "hypothesis": utterance.hypothesis,
                        "teacher_confidence": confidence,
                        "audio_seconds": utterance.audio_seconds,
                        "latency_seconds": utterance.latency_seconds,
                        "speaker": utterance.speaker,
                        "dialect": utterance.dialect,
                        "noise": utterance.noise,
                        "code_switch": utterance.code_switch,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{index}/{len(rows)}] {audio_path.name}", flush=True)
    report = benchmark_report(benchmark_rows)
    report["whisper_basic"] = _whisper_basic_summary(benchmark_rows)
    report["teacher_confidence"] = {
        "mean": statistics.fmean(confidences),
        "p50": statistics.median(confidences),
        "min": min(confidences),
        "max": max(confidences),
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "teacher": teacher.name,
        "model_id": teacher.model_id,
        "model_source": resolved_model_source,
        "family": teacher.family,
        "language_prefix_id": teacher.language_prefix_id,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "split": split,
        "limit": limit,
        "rows": len(rows),
        "predictions": str(predictions_path),
        "metrics": report,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a registered ASR teacher")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-source", default=None, help="Optional local checkpoint path for an already-cached teacher")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    print(
        json.dumps(
            run_benchmark(
                teacher_name=args.teacher,
                manifest=args.manifest,
                output_dir=args.output_dir,
                split=args.split or None,
                limit=args.limit,
                device=args.device,
                model_source=args.model_source,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
