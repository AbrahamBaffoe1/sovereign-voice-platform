"""Guarded NeMo FastPitch training launcher for an explicitly reviewed language configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Accept only an explicit reviewed NeMo config and output directory; language/tokenizer details stay in version-controlled training configuration rather than hidden CLI defaults."""
    parser = argparse.ArgumentParser(description="Train a NeMo FastPitch model from an explicit, reviewed NeMo YAML config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate NeMo dependencies, manifests, and an explicit text tokenizer before launching FastPitch training. The tokenizer gate prevents accidentally training an unsupported language with inherited English assumptions."""
    args = parse_args()
    try:
        import lightning.pytorch as pl
        from nemo.collections.tts.models import FastPitchModel
        from nemo.utils.exp_manager import exp_manager
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("Install NeMo TTS first: pip install -e '.[tts-nemo]'") from exc
    cfg = OmegaConf.load(args.config)
    if "model" not in cfg or "trainer" not in cfg: raise SystemExit("NeMo config must contain top-level model and trainer sections")
    train_manifest = cfg.model.train_ds.dataset.get("manifest_filepath")
    val_manifest = cfg.model.validation_ds.dataset.get("manifest_filepath")
    if not train_manifest or not Path(str(train_manifest)).exists(): raise SystemExit(f"training manifest missing: {train_manifest}")
    if not val_manifest or not Path(str(val_manifest)).exists(): raise SystemExit(f"validation manifest missing: {val_manifest}")
    if cfg.model.get("text_tokenizer") is None: raise SystemExit("Refusing to train without an explicit text_tokenizer. New languages require a reviewed grapheme/phoneme inventory; do not silently inherit English tokenization.")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg.exp_manager = cfg.get("exp_manager", {}); cfg.exp_manager.exp_dir = str(args.output)
    trainer = pl.Trainer(**cfg.trainer); exp_manager(trainer, cfg.exp_manager)
    model = FastPitchModel(cfg=cfg.model, trainer=trainer); trainer.fit(model)
    model.save_to(str(args.output / "fastpitch.nemo"))


if __name__ == "__main__": main()
