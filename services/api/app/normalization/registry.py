"""Registry that maps deployment configuration names to reusable text normalizer instances."""

from __future__ import annotations

from app.normalization.base import TextNormalizer
from app.normalization.ewe import EweNormalizer
from app.normalization.ga import GaNormalizer
from app.normalization.generic import GenericNormalizer
from app.normalization.hausa import HausaNormalizer
from app.normalization.twi import TwiNormalizer


class NormalizerRegistry:
    """Own one stateless normalizer per supported normalization policy.

    Normalizers are intentionally addressed by configuration name rather than language code. That
    keeps routing data-driven and lets multiple language profiles share a policy if future research
    shows that to be safe.
    """

    def __init__(self) -> None:
        """Instantiate the small stateless normalizers once; they are safe to reuse across requests."""
        self._normalizers: dict[str, TextNormalizer] = {
            "generic": GenericNormalizer(),
            "twi": TwiNormalizer(),
            "ewe": EweNormalizer(),
            "ga": GaNormalizer(),
            "hausa": HausaNormalizer(),
        }

    def get(self, name: str) -> TextNormalizer:
        """Return a configured normalizer and conservatively fall back to generic normalization."""
        return self._normalizers.get(name, self._normalizers["generic"])
