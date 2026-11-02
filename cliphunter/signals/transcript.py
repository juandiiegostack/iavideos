"""Whisper transcription and transcript-derived excitement signal."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..cache import Cache
from ..config import Config
from ..exceptions import TranscriptionError
from ..models import SignalSeries, TranscriptData, TranscriptSegment, Word
from ..utils import LOG, moving_average, robust_normalize
from .audio import read_wav_mono
from .base import analysis_times, counts_in_windows

ProgressFn = Callable[[float, str], None]

_MODELS: dict[tuple[str, str, str], Any] = {}


def _norm_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text)


def _resolve_device(config: Config) -> tuple[str, str]:
    requested = str(config.get("transcript.device", "auto") or "auto").lower()
    compute = str(config.get("transcript.compute_type", "auto") or "auto").lower()
    try:
        import ctranslate2

        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        cuda_available = False

    if requested == "auto":
        device = "cuda" if cuda_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        LOG.warning("Se pidió CUDA pero no hay GPU disponible; se usa CPU.")
        device = "cpu"
    else:
        device = requested

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def transcript_cache_name(config: Config) -> str:
    """Cache filename derived from the Whisper model and decoding settings."""
    model_name = str(config.get("transcript.model", "small"))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    payload = {
        "model": model_name,
        "language": config.get("transcript.language"),
        "beam_size": config.i("transcript.beam_size", 5),
        "vad_filter": config.b("transcript.vad_filter", True),
        "condition_on_previous_text": config.b(
            "transcript.condition_on_previous_text", False
        ),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]
    return f"transcript_{safe}_{digest}.json"


def load_cached_transcript(
    cache: Cache, key: str, config: Config
) -> TranscriptData | None:
    """Load a cached transcript without importing or running Whisper."""
    cached = cache.load_json(key, transcript_cache_name(config))
    if not cached:
        return None
    try:
        transcript = TranscriptData.from_dict(cached)
    except (TypeError, ValueError):
        return None
    return transcript if transcript.words else None


def _get_model(config: Config) -> tuple[Any, str, str, str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "Falta faster-whisper. Instala las dependencias con: pip install -r requirements.txt"
        ) from exc

    model_name = str(config.get("transcript.model", "small"))
    device, compute = _resolve_device(config)
    signature = (model_name, device, compute)
    if signature not in _MODELS:
        LOG.info(
            "Cargando Whisper %s en %s (%s). La primera vez descarga el modelo…",
            model_name,
            device,
            compute,
        )
        _MODELS[signature] = WhisperModel(model_name, device=device, compute_type=compute)
    return _MODELS[signature], model_name, device, compute


def transcribe(
    audio_path: str | Path,
    cache: Cache,
    key: str,
    config: Config,
    duration: float,
    *,
    on_progress: ProgressFn | None = None,
    force: bool = False,
) -> TranscriptData:
    """Transcribe with faster-whisper and cache word-level timestamps."""
    cache_name = transcript_cache_name(config)
    if not force:
        transcript = load_cached_transcript(cache, key, config)
        if transcript:
            LOG.info(
                "Transcripción en caché: %d palabras (%s)",
                len(transcript.words),
                transcript.language or "idioma ?",
            )
            return transcript

    # Pass a NumPy array instead of a path: recent faster-whisper versions call
    # av.open(..., metadata_errors=...), which breaks with some PyAV builds.
    audio, sample_rate = read_wav_mono(audio_path)
    if sample_rate != 16000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate), endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    language = config.get("transcript.language") or None
    last_error: Exception | None = None
    for attempt in range(2):
        model, model_name, device, compute = _get_model(config)
        try:
            segments_iter, info = model.transcribe(
                audio,
                language=language,
                beam_size=config.i("transcript.beam_size", 5),
                vad_filter=config.b("transcript.vad_filter", True),
                word_timestamps=True,
                condition_on_previous_text=config.b(
                    "transcript.condition_on_previous_text", False
                ),
            )
            words: list[Word] = []
            segments: list[TranscriptSegment] = []
            detected = getattr(info, "language", None) or language
            for segment in segments_iter:
                text = (segment.text or "").strip()
                if text:
                    segments.append(
                        TranscriptSegment(float(segment.start), float(segment.end), text)
                    )
                for word in segment.words or []:
                    token = (word.word or "").strip()
                    if token:
                        words.append(
                            Word(
                                start=float(word.start),
                                end=float(word.end),
                                text=token,
                                probability=float(getattr(word, "probability", 0.0) or 0.0),
                            )
                        )
                if on_progress:
                    fraction = min(0.99, float(segment.end) / duration) if duration else 0.0
                    on_progress(fraction, f"Transcribiendo {fraction * 100:.0f}%")

            transcript = TranscriptData(
                words=words,
                segments=segments,
                language=detected,
                duration=duration,
                model=f"{model_name}/{device}/{compute}",
            )
            if on_progress:
                on_progress(1.0, "Transcripción completada")
            cache.save_json(key, cache_name, transcript.to_dict())
            if words:
                LOG.info(
                    "Transcripción lista: %d palabras, idioma %s",
                    len(words),
                    transcript.language or "?",
                )
            else:
                LOG.warning("Whisper no detectó voz en el audio.")
            return transcript
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0 and device == "cuda":
                LOG.warning("Whisper falló en GPU (%s); reintentando en CPU.", exc)
                config.set("transcript.device", "cpu")
                config.set("transcript.compute_type", "int8")
                continue
            break

    raise TranscriptionError(f"Whisper no pudo transcribir el audio: {last_error}")


def _keyword_map(config: Config) -> dict[int, dict[str, str]]:
    keyword_config = config.get("transcript.keywords", {}) or {}
    phrases: list[str] = []
    if isinstance(keyword_config, dict):
        for values in keyword_config.values():
            for value in values or []:
                if str(value).strip():
                    phrases.append(str(value))
    elif isinstance(keyword_config, list):
        phrases.extend(str(v) for v in keyword_config if str(v).strip())

    result: dict[int, dict[str, str]] = {}
    seen: set[str] = set()
    for phrase in phrases:
        tokens = [_norm_token(t) for t in phrase.split()]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        key = " ".join(tokens)
        if key in seen:
            continue
        seen.add(key)
        # First spelling wins, so accented forms ("increíble") are preserved.
        result.setdefault(len(tokens), {})[key] = phrase
    return result


def analyze_transcript(
    transcript: TranscriptData | None,
    duration: float,
    config: Config,
) -> SignalSeries:
    """Score keyword hits, speech-rate bursts and exclamations."""
    window = config.f("analysis.window", 10.0)
    hop = config.f("analysis.hop", 2.0)
    times = analysis_times(duration, hop)

    if transcript is None or not transcript.words:
        return SignalSeries(
            "transcript", times, np.zeros(len(times), dtype=np.float32), False,
            "sin transcripción",
        )

    words = [w for w in transcript.words if w.end >= 0]
    tokens = [_norm_token(w.text) for w in words]
    by_len = _keyword_map(config)
    max_parts = max(by_len) if by_len else 1

    hit_times: list[float] = []
    hit_weights: list[float] = []
    hit_labels: list[tuple[float, str, float]] = []
    for index, _token in enumerate(tokens):
        for parts in range(1, max_parts + 1):
            if index + parts > len(tokens):
                break
            phrase_tokens = tokens[index : index + parts]
            if any(not t for t in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            label = by_len.get(parts, {}).get(phrase)
            if label:
                weight = 1.0 + 0.25 * (parts - 1)
                hit_times.append(words[index].start)
                hit_weights.append(weight)
                hit_labels.append((words[index].start, label, weight))

    word_times = np.array([w.start for w in words], dtype=np.float64)
    speech_rate = counts_in_windows(word_times, times, window) / window
    keyword_rate = counts_in_windows(
        np.array(hit_times, dtype=np.float64), times, window,
        np.array(hit_weights, dtype=np.float64),
    )
    baseline_span = max(1, int(round(60.0 / hop)))
    baseline = moving_average(speech_rate, baseline_span)
    rate_spike = np.maximum(0.0, speech_rate - baseline) / (baseline + 0.5)

    raw = (
        0.60 * robust_normalize(keyword_rate)
        + 0.25 * robust_normalize(rate_spike)
        + 0.15 * robust_normalize(speech_rate)
    )
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)

    note = f"{len(words)} palabras"
    if hit_labels:
        note += f", {len(hit_labels)} palabras clave"
    LOG.info("Señal de transcripción: %s", note)

    return SignalSeries(
        "transcript",
        times,
        raw,
        True,
        note,
        detail={
            "keyword_rate": keyword_rate,
            "rate_spike": rate_spike,
            "speech_rate": speech_rate,
            "keyword_hits": hit_labels,
        },
    )
