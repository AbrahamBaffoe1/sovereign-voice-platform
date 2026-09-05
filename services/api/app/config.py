"""Validated runtime configuration loaded from VOICE_* environment variables and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single validated source of runtime configuration. Keeping model, networking, limits, and storage
    settings here prevents hidden constants from spreading through engine code."""
    model_config = SettingsConfigDict(
        env_prefix="VOICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: str | None = None
    log_level: str = "INFO"

    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")
    language_config: Path = Path("./config/languages.yaml")

    asr_engine: Literal["faster_whisper"] = "faster_whisper"
    asr_model: str = "large-v3"
    asr_device: str = "auto"
    asr_compute_type: str = "default"
    asr_beam_size: int = Field(default=5, ge=1, le=20)
    asr_vad: bool = True

    llm_enabled: bool = True
    llm_base_url: str = "http://127.0.0.1:8001/v1"
    llm_model: str = "local-model"
    llm_api_key: str = "local"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=512, ge=32, le=8192)
    llm_temperature: float = Field(default=0.35, ge=0.0, le=2.0)

    default_tts_engine: Literal["chatterbox", "nemo", "voxcpm"] = "chatterbox"
    chatterbox_device: str = "auto"
    chatterbox_model: str = "v3"
    nemo_device: str = "auto"
    voxcpm_device: str = "auto"
    voxcpm_cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    voxcpm_inference_timesteps: int = Field(default=10, ge=1, le=50)

    max_upload_mb: int = Field(default=30, ge=1, le=500)
    max_ws_audio_seconds: int = Field(default=90, ge=5, le=600)
    max_corpus_clip_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    max_corpus_recording_seconds: float = Field(default=1800.0, ge=30.0, le=7200.0)
    default_sample_rate: int = Field(default=16000, ge=8000, le=48000)

    @property
    def max_upload_bytes(self) -> int:
        """Convert the human-friendly megabyte setting into the byte limit used by upload readers."""
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache one Settings object per process, then create the minimum local storage
    directories required by runtime services. Caching ensures dependency construction sees one
    consistent configuration snapshot."""
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "voices").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "corpus").mkdir(parents=True, exist_ok=True)
    return settings
