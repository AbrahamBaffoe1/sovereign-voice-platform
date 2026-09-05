"""Guarded NeMo HiFi-GAN launcher for the vocoder stage of a custom-language TTS stack."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Require a version-controlled NeMo config and explicit output directory."""
    parser = argparse.ArgumentParser(description="Train or fine-tune NeMo HiFi-GAN from YAML")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate experiment structure, run Lightning training, and persist a serving checkpoint."""
    args = parse_args()
    try:
        import lightning.pytorch as pl
        from nemo.collections.tts.models import HifiGanModel
        from nemo.utils.exp_manager import exp_manager
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("Install NeMo TTS first: pip install -e '.[tts-nemo]'") from exc
    cfg = OmegaConf.load(args.config)
    if "model" not in cfg or "trainer" not in cfg:
        raise SystemExit("NeMo config must contain top-level model and trainer sections")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg.exp_manager = cfg.get("exp_manager", {})
    cfg.exp_manager.exp_dir = str(args.output)
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)
    model.save_to(str(args.output / "hifigan.nemo"))


if __name__ == "__main__":
    main()
