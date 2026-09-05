"""Whisper fine-tuning entry point supporting explicit or token-free low-resource language experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.common.language_profile import load_language_profile


def parse_args() -> argparse.Namespace:
    """Expose reproducible training knobs and an optional language profile as the source of defaults."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a local speech corpus")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
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
    """Prepare local manifests, fine-tune Whisper, evaluate WER and save model plus processor together."""
    args = parse_args()
    experiment = resolve_experiment(args)
    try:
        import evaluate
        from datasets import Audio, DatasetDict, load_dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise SystemExit("Install training ASR dependencies: pip install -e '.[training-asr]'") from exc

    data = load_dataset(
        "json",
        data_files={"train": str(args.train), "validation": str(args.validation)},
    )
    assert isinstance(data, DatasetDict)

    def expose_audio(row: dict[str, Any]) -> dict[str, Any]:
        """Expose manifest paths through datasets.Audio so decoding/resampling remains lazy."""
        row["audio"] = row["audio_filepath"]
        return row

    data = data.map(expose_audio)
    data = data.cast_column("audio", Audio(sampling_rate=16000))

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
        """Convert one waveform into Whisper log-mel features and tokenize the reviewed transcript."""
        audio = row["audio"]
        row["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        row["labels"] = processor.tokenizer(str(row["text"])).input_ids
        return row

    remove_columns = data["train"].column_names
    data = data.map(prepare, remove_columns=remove_columns, num_proc=1)
    metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        """Decode generated predictions and labels and return WER percentage for checkpoint selection."""
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
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=min(500, max(50, args.max_steps // 10)),
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=3,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    final = args.output / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    (final / "experiment.json").write_text(
        __import__("json").dumps(
            {
                "base_model": experiment.base_model,
                "language_token_mode": experiment.language_token_mode,
                "decoder_language": experiment.decoder_language,
                "profile": str(args.profile) if args.profile else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
