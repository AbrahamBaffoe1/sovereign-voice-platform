"""Whisper fine-tuning entry point for local speech manifests, including padding and WER evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Define the reproducible training knobs exposed at the command line. The decoder language is
    mandatory because silently guessing a tokenizer language for unsupported speech would invalidate
    experiments."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a local speech corpus")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="openai/whisper-small")
    parser.add_argument("--decoder-language", required=True)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Batch collator that pads audio features and decoder labels independently, masks label padding
    from the loss, and removes the duplicated decoder-start token expected by Whisper training."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one padded training batch from variable-length processed examples and convert
        tokenizer padding positions to -100 so PyTorch cross-entropy ignores them."""
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    """Load speech manifests, decode/resample audio, build Whisper features and labels, configure
    deterministic evaluation/checkpoint cadence, train with Seq2SeqTrainer, and save the final model
    plus processor together."""
    args = parse_args()
    try:
        import evaluate
        from datasets import Audio, DatasetDict, load_dataset
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise SystemExit("Install the training-asr extra first: pip install -e '.[training-asr]'") from exc

    data = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.validation)})
    assert isinstance(data, DatasetDict)

    def expose_audio(row: dict[str, Any]) -> dict[str, Any]:
        """Mirror the manifest audio_filepath field into the datasets library audio column so Hugging
        Face can decode and resample the file lazily."""
        row["audio"] = row["audio_filepath"]
        return row

    data = data.map(expose_audio)
    data = data.cast_column("audio", Audio(sampling_rate=16000))
    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.decoder_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.config.use_cache = False
    model.generation_config.language = args.decoder_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        """Convert one decoded waveform into Whisper log-mel input features and tokenize its transcript
        into decoder labels."""
        audio = row["audio"]
        row["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        row["labels"] = processor.tokenizer(str(row["text"])).input_ids
        return row

    remove_columns = data["train"].column_names
    data = data.map(prepare, remove_columns=remove_columns, num_proc=1)
    metric = evaluate.load("wer")

    def compute_metrics(pred: Any) -> dict[str, float]:
        """Decode generated predictions and masked labels back to text and report WER as a percentage
        for checkpoint selection."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100.0 * float(metric.compute(predictions=pred_text, references=label_text))}

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(output_dir=str(args.output),per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=max(1,args.batch_size//2),gradient_accumulation_steps=args.gradient_accumulation,learning_rate=args.learning_rate,warmup_steps=min(500,max(50,args.max_steps//10)),max_steps=args.max_steps,gradient_checkpointing=True,fp16=args.fp16,bf16=args.bf16,eval_strategy="steps",eval_steps=250,save_steps=250,logging_steps=25,predict_with_generate=True,generation_max_length=225,load_best_model_at_end=True,metric_for_best_model="wer",greater_is_better=False,save_total_limit=3,report_to=["tensorboard"],remove_unused_columns=False)
    trainer = Seq2SeqTrainer(model=model,args=training_args,train_dataset=data["train"],eval_dataset=data["validation"],data_collator=collator,compute_metrics=compute_metrics,processing_class=processor)
    trainer.train()
    trainer.save_model(str(args.output / "final"))
    processor.save_pretrained(str(args.output / "final"))


if __name__ == "__main__":
    main()
