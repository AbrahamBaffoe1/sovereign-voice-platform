"""Registry that maps configuration names to reusable text normalizer instances."""

from __future__ import annotations

from app.normalization.base import TextNormalizer
from app.normalization.generic import GenericNormalizer
from app.normalization.twi import TwiNormalizer


class NormalizerRegistry:
    """Small in-memory registry of stateless normalizers keyed by the names used in languages.yaml."""
    def __init__(self) -> None:
        """Instantiate normalizers once because they are stateless and safe to share across requests."""
        self._normalizers: dict[str, TextNormalizer] = {
            "generic": GenericNormalizer(),
            "twi": TwiNormalizer(),
        }

    def get(self, name: str) -> TextNormalizer:
        """Return the requested normalizer and fall back to the conservative generic normalizer when
        configuration names a future/unknown normalizer."""
        return self._normalizers.get(name, self._normalizers["generic"])
