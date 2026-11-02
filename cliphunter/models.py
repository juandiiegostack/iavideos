"""Dataclasses shared by the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MediaInfo:
    """Technical metadata for the analyzed media file."""

    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    title: str = ""
    source_url: str = ""
    video_id: str = ""
    platform: str = ""

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.path),
            "duration": round(self.duration, 2),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "has_audio": self.has_audio,
            "title": self.title,
            "source_url": self.source_url,
            "video_id": self.video_id,
            "platform": self.platform,
        }


@dataclass
class ChatMessage:
    """Normalized chat message with time relative to the source start."""

    t: float
    text: str
    emotes: list[str] = field(default_factory=list)
    author: str = ""


@dataclass
class Word:
    """A single transcribed word with timestamps in source-media seconds."""

    start: float
    end: float
    text: str
    probability: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "probability": round(self.probability, 3),
        }


@dataclass
class TranscriptSegment:
    """A Whisper segment (used for the transcript excerpt)."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


@dataclass
class TranscriptData:
    """Full transcription with word-level timings."""

    words: list[Word] = field(default_factory=list)
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0
    model: str = ""

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments).strip()

    def excerpt(self, start: float, end: float, max_chars: int = 280) -> str:
        """Return the transcript text overlapping ``[start, end]``."""
        words = [w.text for w in self.words if w.end > start and w.start < end]
        text = " ".join(words).strip()
        if not text:
            text = " ".join(
                s.text for s in self.segments if s.end > start and s.start < end
            ).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration": round(self.duration, 3),
            "model": self.model,
            "words": [w.to_dict() for w in self.words],
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptData":
        return cls(
            words=[Word(**w) for w in data.get("words", [])],
            segments=[TranscriptSegment(**s) for s in data.get("segments", [])],
            language=data.get("language"),
            duration=float(data.get("duration", 0.0)),
            model=data.get("model", ""),
        )


@dataclass
class SignalSeries:
    """A time series produced by one signal, sampled on the analysis grid.

    ``times`` are window centers in seconds and are shared by all signals.
    ``detail`` may contain auxiliary numpy arrays used to explain a clip.
    """

    name: str
    times: np.ndarray
    values: np.ndarray
    available: bool = True
    note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    envelope_times: np.ndarray | None = None
    envelope_db: np.ndarray | None = None

    def resampled(self, target_times: np.ndarray) -> np.ndarray:
        """Linearly resample this signal onto ``target_times`` (clamped)."""
        if len(self.times) == 0 or len(self.values) == 0:
            return np.zeros(len(target_times), dtype=np.float32)
        if np.array_equal(self.times, target_times):
            return self.values.astype(np.float32, copy=False)
        return np.interp(
            target_times, self.times, self.values, left=self.values[0], right=self.values[-1]
        ).astype(np.float32)

    def to_meta_dict(self) -> dict[str, Any]:
        """JSON-safe metadata (the raw arrays are cached separately)."""
        return {
            "name": self.name,
            "available": self.available,
            "note": self.note,
            "frames": int(len(self.values)),
        }


@dataclass
class CandidateClip:
    """One selected moment plus the evidence that selected it."""

    index: int
    start: float
    end: float
    peak: float
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    title: str = ""
    transcript: str = ""
    llm_score: float | None = None
    llm_reason: str = ""
    file: str | None = None
    subtitle_file: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "file": self.file,
            "subtitle_file": self.subtitle_file,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "peak": round(self.peak, 2),
            "score": round(self.score, 4),
            "signals": {k: round(v, 4) for k, v in self.breakdown.items()},
            "reasons": self.reasons,
            "title": self.title,
            "transcript": self.transcript,
            "llm_score": self.llm_score,
            "llm_reason": self.llm_reason,
        }


@dataclass
class AnalysisResult:
    """Output of the scoring stage."""

    times: np.ndarray
    scores: np.ndarray
    signals: dict[str, SignalSeries]
    clips: list[CandidateClip]
    transcript: TranscriptData | None = None
