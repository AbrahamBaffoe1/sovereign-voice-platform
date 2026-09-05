"""Stable domain-level exception types used to keep engine failures separate from transport-specific HTTP handling."""

class VoicePlatformError(Exception):
    """Base class for errors that can be safely mapped to an API response."""


class ConfigurationError(VoicePlatformError):
    """Raised when deployment configuration is structurally valid Python but cannot produce a usable
    runtime."""
    pass


class EngineUnavailableError(VoicePlatformError):
    """Raised when an optional model backend or its Python dependency is not installed or available."""
    pass


class UnsupportedLanguageError(VoicePlatformError):
    """Raised when a requested language code is absent from the deployment language registry."""
    pass


class InvalidAudioError(VoicePlatformError):
    """Raised when audio violates a boundary invariant before or after model inference."""
    pass


class InvalidRequestError(VoicePlatformError, ValueError):
    """Raised when caller-provided domain data is well-formed at transport level but violates a business rule."""
    pass


class ConflictError(VoicePlatformError, ValueError):
    """Raised when a valid request conflicts with current durable state, such as a duplicate corpus item or stale review transition."""
    pass


class ResourceNotFoundError(VoicePlatformError):
    """Raised when a caller references a local resource such as a voice profile that does not exist."""
    pass


class ModelInferenceError(VoicePlatformError):
    """Raised when a configured model exists but fails while performing inference."""
    pass
