"""Typed errors so the CLI can show actionable messages instead of tracebacks."""

from __future__ import annotations


class ClipHunterError(RuntimeError):
    """Base error for all expected ClipHunter failures."""


class DependencyError(ClipHunterError):
    """A required external tool or Python package is missing."""


class SourceError(ClipHunterError):
    """The source URL/file is invalid or cannot be read."""


class UnsupportedSourceError(SourceError):
    """The URL is not a supported VOD/live source."""


class LiveOfflineError(SourceError):
    """The requested channel is not live."""


class ChatUnavailable(ClipHunterError):
    """Chat could not be downloaded for this source."""


class TranscriptionError(ClipHunterError):
    """Whisper failed or produced no usable output."""


class LLMError(ClipHunterError):
    """The optional LLM step failed."""
