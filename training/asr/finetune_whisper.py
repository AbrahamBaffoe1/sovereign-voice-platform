"""Whisper fine-tuning entry point supporting explicit or token-free low-resource language experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from training.common.language_profile import load_language_profile
from training.common.manifest import file_sha256


def parse_args() -> argparse.Namespace:
    """Expose reproducible training knobs and an optional language profile as the source of defaults."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a local speech corpus")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--validation",
        type=Path,
        default=None,
        help="Optional internal validation manifest. Omit it rather than reusing the external release benchmark.",
    )
    parser.add_argument("--dataset-version", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--decoder-language",
        default=None,
        help="Only set when the chosen Whisper tokenizer strategy has a reviewed language token.",
    )
    parser.add_argument(
        "--language-token-mode",
        choices=("none", "explicit"),
        default=None,
        help="Override profile ASR token policy. 'none' avoids inventing a language token.",
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Use 'auto' for the highest checkpoint-* under --output, or pass an explicit checkpoint directory.",
    )
    return parser.parse_args()


@dataclass(frozen=True, slots=True)
class ResolvedASRExperiment:
    """Fully resolved tokenizer/model choices recorded before GPU training begins."""

    base_model: str
    language_token_mode: str
    decoder_language: str | None


def resolve_experiment(args: argparse.Namespace) -> ResolvedASRExperiment:
    """Merge CLI overrides with a language profile and reject contradictory decoder-token settings."""
    profile = load_language_profile(args.profile) if args.profile else None
    base_model = args.base_model or (profile.asr.base_model if profile else "openai/whisper-small")
    mode = args.language_token_mode or (profile.asr.language_token_mode if profile else None)
    decoder_language = args.decoder_language
    if decoder_language is None and profile:
        decoder_language = profile.asr.decoder_language
    if mode is None:
        mode = "explicit" if decoder_language else "none"
    if mode == "explicit" and not decoder_language:
        raise SystemExit("explicit language-token mode requires --decoder-language")
    if mode == "none" and decoder_language:
        raise SystemExit("decoder language must be omitted when language-token mode is 'none'")
    return ResolvedASRExperiment(base_model, mode, decoder_language)


def _jsonl_rows(path: Path | None) -> int:
    """Count non-empty manifest rows without loading an entire corpus into memory."""
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _dataset_lineage(args: argparse.Namespace) -> tuple[Path | None, dict[str, Any] | None]:
    """Load the frozen dataset identity that must accompany every trained checkpoint."""
    version_path = args.dataset_version
    if version_path is None:
        candidate = args.train.parent / "dataset_version.json"
        version_path = candidate if candidate.exists() else None
    if version_path is None:
        return None, None
    payload = json.loads(version_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid dataset version file: {version_path}")
    if int(payload.get("accepted", 0)) < 1:
        raise SystemExit(f"refusing to train on an empty frozen corpus: {version_path}")
    return version_path, payload


def _latest_checkpoint(output: Path) -> Path | None:
    """Return the checkpoint with the highest trainer step, ignoring unrelated directories in the run folder."""
    best: tuple[int, Path] | None = None
    if not output.exists():
        return None
    for path in output.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if best is None or step > best[0]:
            best = (step, path)
    return best[1] if best else None


def _resolve_resume_checkpoint(output: Path, requested: str | None) -> str | None:
    """Resolve resume intent before model loading so bad checkpoint paths fail without spending GPU memory."""
    if requested is None:
        return None
    if requested == "auto":
        latest = _latest_checkpoint(output)
        return str(latest) if latest else None
    path = Path(requested).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"resume checkpoint does not exist: {path}")
    return str(path)


def _read_frozen_wav(path: str | Path, *, required_sample_rate: int = 16000) -> np.ndarray:
    """Read the compiler-owned WAV directly and re-check the ASR audio contract at the training boundary."""
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != required_sample_rate:
        raise ValueError(f"{path}: expected {required_sample_rate} Hz, found {sample_rate} Hz")
    if waveform.ndim != 1:
        raise ValueError(f"{path}: expected mono audio, found shape {waveform.shape}")
    if not np.isfinite(waveform).all():
        raise ValueError(f"{path}: waveform contains non-finite samples")
    return np.asarray(waveform, dtype=np.float32)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad acoustic features and decoder labels independently while masking label padding from loss."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one batch and remove a duplicated decoder-start token when every example contains it."""
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    """Prepare frozen manifests, fine-tune Whisper, and save model/processor plus complete lineage."""
    args = parse_args()
    if args.fp16 and args.bf16:
        raise SystemExit("choose at most one of --fp16 and --bf16")
    if args.max_steps < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("max steps, batch size and gradient accumulation must be positive")
    train_rows = _jsonl_rows(args.train)
    if train_rows < 1:
        raise SystemExit(f"training manifest is empty or missing: {args.train}")
    validation_rows = _jsonl_rows(args.validation)
    validation_path = args.validation if validation_rows else None
    version_path, dataset_version = _dataset_lineage(args)
    experiment = resolve_experiment(args)
    resume_checkpoint = _resolve_resume_checkpoint(args.output, args.resume_from_checkpoint)

    try:
        import evaluate
        from datasets import DatasetDict, load_dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise SystemExit("Install training ASR dependencies: pip install -e '.[training-asr]'") from exc

    data_files = {"train": str(args.train)}
    if validation_path is not None:
        data_files["validation"] = str(validation_path)
    data = load_dataset("json", data_files=data_files)
    assert isinstance(data, DatasetDict)

    if experiment.language_token_mode == "explicit":
        processor = WhisperProcessor.from_pretrained(
            experiment.base_model,
            language=experiment.decoder_language,
            task="transcribe",
        )
    else:
        processor = WhisperProcessor.from_pretrained(experiment.base_model)

    model = WhisperForConditionalGeneration.from_pretrained(experiment.base_model)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    if experiment.language_token_mode == "explicit":
        model.generation_config.language = experiment.decoder_language
        model.generation_config.task = "transcribe"

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        """Convert one compiler-normalized WAV into Whisper log-mel features and reviewed transcript tokens."""
        # corpus-v0 already owns decoding/resampling. Reading the local WAV directly keeps training
        # independent of Hugging Face Audio/TorchCodec representation changes and validates the freeze.
        waveform = _read_frozen_wav(str(row["audio_filepath"]))
        row["input_features"] = processor.feature_extractor(
            waveform, sampling_rate=16000
        ).input_features[0]
        row["labels"] = processor.tokenizer(str(row["text"])).input_ids
        return row

    remove_columns = data["train"].column_names
    data = data.map(prepare, remove_columns=remove_columns, num_proc=1)
    metric = evaluate.load("wer") if validation_path is not None else None

    def compute_metrics(pred: Any) -> dict[str, float]:
        """Decode generated predictions and labels and return WER percentage for checkpoint selection."""
        assert metric is not None
        pred_ids = pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100.0 * float(metric.compute(predictions=pred_text, references=label_text))}

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    has_validation = validation_path is not None
    training_kwargs: dict[str, Any] = {
        "output_dir": str(args.output),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": max(1, args.batch_size // 2),
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "warmup_steps": min(500, max(50, args.max_steps // 10)),
        "max_steps": args.max_steps,
        "gradient_checkpointing": True,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "eval_strategy": "steps" if has_validation else "no",
        "save_strategy": "steps",
        "save_steps": 250,
        "logging_steps": 25,
        "predict_with_generate": has_validation,
        "generation_max_length": 225,
        "load_best_model_at_end": has_validation,
        "save_total_limit": 3,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
    }
    if has_validation:
        training_kwargs.update(
            {
                "eval_steps": 250,
                "metric_for_best_model": "wer",
                "greater_is_better": False,
            }
        )
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"] if has_validation else None,
        data_collator=collator,
        compute_metrics=compute_metrics if has_validation else None,
        processing_class=processor,
    )

    # Trainer checkpoints contain optimizer/scheduler/RNG state, not just model weights. Passing the
    # selected directory back to Trainer is what makes a resumed run numerically continue the old run.
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final = args.output / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    (final / "experiment.json").write_text(
        json.dumps(
            {
                "base_model": experiment.base_model,
                "language_token_mode": experiment.language_token_mode,
                "decoder_language": experiment.decoder_language,
                "profile": str(args.profile) if args.profile else None,
                "profile_sha256": file_sha256(args.profile) if args.profile else None,
                "train_manifest": str(args.train),
                "train_manifest_sha256": file_sha256(args.train),
                "train_rows": train_rows,
                "validation_manifest": str(validation_path) if validation_path else None,
                "validation_manifest_sha256": file_sha256(validation_path) if validation_path else None,
                "validation_rows": validation_rows,
                "selection_policy": "best_validation_wer" if has_validation else "fixed_steps_final",
                "dataset_version": str(version_path) if version_path else None,
                "dataset_fingerprint_sha256": (
                    dataset_version.get("fingerprint_sha256") if dataset_version else None
                ),
                "resume_requested": args.resume_from_checkpoint,
                "resumed_from_checkpoint": resume_checkpoint,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate,
                "fp16": args.fp16,
                "bf16": args.bf16,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
