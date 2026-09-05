"""Safe wrapper around the CTranslate2 Whisper converter used for Faster-Whisper deployment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    """Validate converter availability and output safety, then invoke the official CTranslate2
    converter without shell interpolation. Refusing to overwrite a non-empty directory protects
    previous deployable checkpoints."""
    parser = argparse.ArgumentParser(description="Export a fine-tuned Whisper model to CTranslate2")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        default="float16",
        choices=["float16", "float32", "int8", "int8_float16", "int8_float32"],
    )
    args = parser.parse_args()

    converter = shutil.which("ct2-transformers-converter")
    if not converter:
        raise SystemExit(
            "ct2-transformers-converter not found. Install faster-whisper/CTranslate2 tooling first."
        )
    if not args.model.exists():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    command = [
        converter,
        "--model",
        str(args.model),
        "--output_dir",
        str(args.output),
        "--quantization",
        args.quantization,
        "--copy_files",
        "tokenizer.json",
        "preprocessor_config.json",
        "generation_config.json",
    ]
    subprocess.run(command, check=True)
    print(f"exported CTranslate2 model to {args.output}")
    print(f"set VOICE_ASR_MODEL={args.output}")


if __name__ == "__main__":
    main()
