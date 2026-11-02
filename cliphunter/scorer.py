"""Combine normalized signals, find peaks and select non-overlapping clips."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from .config import Config
from .models import AnalysisResult, CandidateClip, SignalSeries, TranscriptData
from .signals.base import analysis_times
from .utils import LOG, moving_average, robust_normalize


def _clip_window(
    peak: float,
    duration: float,
    config: Config,
) -> tuple[float, float]:
    """Build a window around a peak respecting pre/post roll and min/max length."""
    pre = config.f("selection.pre_roll", 8.0)
    post = config.f("selection.post_roll", 4.0)
    min_dur = config.f("selection.min_duration", 15.0)
    max_dur = config.f("selection.max_duration", 60.0)

    if duration <= min_dur:
        return 0.0, duration

    start = peak - pre
    end = peak + post
    if end - start < min_dur:
        end += min_dur - (end - start)
    if end - start > max_dur:
        start = peak - max_dur / 2.0
        end = start + max_dur

    if start < 0:
        end = min(duration, end - start)
        start = 0.0
    if end > duration:
        start = max(0.0, start - (end - duration))
        end = duration

    if end - start < min_dur:
        if start <= 1e-6:
            end = min(duration, min_dur)
        else:
            start = max(0.0, end - min_dur)
    return max(0.0, start), min(duration, end)


def _snap_to_silence(
    start: float,
    end: float,
    peak: float,
    duration: float,
    signal: SignalSeries | None,
    config: Config,
) -> tuple[float, float]:
    """Move clip boundaries to nearby quiet points when that stays valid."""
    if signal is None or signal.envelope_times is None or signal.envelope_db is None:
        return start, end
    env_t = np.asarray(signal.envelope_times, dtype=np.float64)
    env_db = np.asarray(signal.envelope_db, dtype=np.float64)
    if len(env_t) < 5:
        return start, end

    tol = config.f("selection.silence_snap", 3.0)
    quiet_db = config.f("audio.quiet_db", -42.0)
    min_dur = config.f("selection.min_duration", 15.0)
    max_dur = config.f("selection.max_duration", 60.0)

    local_min = (env_db <= np.roll(env_db, 1)) & (env_db <= np.roll(env_db, -1))
    local_min[0] = local_min[-1] = False
    quiet = local_min & (env_db <= quiet_db)
    candidates = np.flatnonzero(quiet)
    if len(candidates) == 0:
        candidates = np.flatnonzero(local_min)
    if len(candidates) == 0:
        return start, end

    def nearest(target: float) -> float | None:
        times = env_t[candidates]
        within = np.abs(times - target) <= tol
        if not np.any(within):
            return None
        options = times[within]
        return float(options[np.argmin(np.abs(options - target))])

    new_start = nearest(start)
    if new_start is not None and new_start <= peak - 2.0 and end - new_start >= min_dur:
        start = max(0.0, new_start)
    new_end = nearest(end)
    if new_end is not None and new_end >= peak + 2.0 and new_end - start >= min_dur:
        end = min(duration, new_end)
    if end - start > max_dur:
        start, end = _clip_window(peak, duration, config)
    return start, end


def _window_slice(times: np.ndarray, start: float, end: float) -> slice:
    left = int(np.searchsorted(times, start, side="left"))
    right = int(np.searchsorted(times, end, side="right"))
    if right <= left:
        return slice(max(0, left - 1), min(len(times), left + 1))
    return slice(left, right)


def _fallback_title(
    rank: int,
    start: float,
    end: float,
    transcript: TranscriptData | None,
    transcript_signal: SignalSeries | None,
) -> str:
    """Use the strongest keyword inside the clip, or a generic title."""
    if transcript_signal is not None and transcript_signal.available:
        hits = transcript_signal.detail.get("keyword_hits", [])
        inside = [(t, label, w) for (t, label, w) in hits if start <= t <= end]
        if inside:
            labels = [label for _, label, _ in sorted(inside, key=lambda x: -x[2])]
            label = labels[0]
            return label[:1].upper() + label[1:70]
    if transcript is not None:
        text = transcript.excerpt(start, end, 45).strip(" .,;:!?…")
        if text:
            text = text[:1].upper() + text[1:]
            if len(text) > 60:
                text = text[:57].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"
            return text
    return f"Momento destacado {rank}"


def _reasons(
    start: float,
    end: float,
    times: np.ndarray,
    normalized: dict[str, np.ndarray],
    signals: dict[str, SignalSeries],
    transcript: TranscriptData | None,
    config: Config,
) -> tuple[list[str], dict[str, float]]:
    """Explain which signals fired inside a selected window."""
    slc = _window_slice(times, start, end)
    breakdown: dict[str, float] = {}
    reasons: list[str] = []
    window = config.f("analysis.window", 10.0)

    for name, values in normalized.items():
        if len(values) == 0:
            continue
        breakdown[name] = float(np.max(values[slc])) if len(values[slc]) else 0.0

    chat = signals.get("chat")
    if chat is not None and chat.name in breakdown and breakdown["chat"] > 0.15:
        detail = chat.detail
        rate = detail.get("rate")
        spike = detail.get("spike")
        density = detail.get("emote_density")
        if rate is not None and len(rate):
            rate_max = float(np.max(rate[slc]))
            baseline = float(np.median(rate)) or 0.0
            ratio = rate_max / baseline if baseline > 0.01 else 0.0
            if ratio >= 1.6:
                reasons.append(
                    f"chat: {ratio:.1f}× el promedio ({rate_max * window:.0f} msg/{window:.0f}s)"
                )
            elif spike is not None and len(spike) and float(np.max(spike[slc])) > 0.8:
                reasons.append("chat: pico de actividad")
        if density is not None and len(density) and float(np.max(density[slc])) > 0.35:
            reasons.append("chat: pico de emotes")
        if not reasons:
            reasons.append("chat: actividad destacada")

    audio = signals.get("audio")
    if audio is not None and audio.name in breakdown and breakdown["audio"] > 0.25:
        detail = audio.detail
        db = detail.get("rms_db")
        scream = detail.get("scream")
        laughter = detail.get("laughter")
        novelty = detail.get("novelty")
        if scream is not None and len(scream) and float(np.max(scream[slc])) > 0.22:
            reasons.append("audio: grito / pico agudo")
        if laughter is not None and len(laughter) and float(np.max(laughter[slc])) > 0.03:
            reasons.append("audio: risas")
        if novelty is not None and len(novelty) and float(np.max(novelty[slc])) > 7.0:
            reasons.append("audio: subida brusca de volumen")
        if db is not None and len(db):
            peak_db = float(np.max(db[slc]))
            if not any(r.startswith("audio:") for r in reasons):
                reasons.append(f"audio: pico de volumen ({peak_db:.0f} dB)")

    if "transcript" in breakdown and breakdown["transcript"] > 0.2:
        hits: list[tuple[float, str, float]] = []
        if "transcript" in signals:
            hits = signals["transcript"].detail.get("keyword_hits", [])
        inside = [(t, label, w) for (t, label, w) in hits if start <= t <= end]
        if inside:
            counts: Counter[str] = Counter()
            for _, label, weight in inside:
                counts[label] += weight
            labels = [label for label, _ in counts.most_common(3)]
            quoted = ", ".join(f"«{label}»" for label in labels)
            reasons.append(f"transcripción: {quoted}")
        else:
            reasons.append("transcripción: habla acelerada")
    return reasons, breakdown


def score_and_select(
    signals: dict[str, SignalSeries],
    transcript: TranscriptData | None,
    duration: float,
    config: Config,
) -> AnalysisResult:
    """Normalize signals, combine them, and pick the best non-overlapping clips."""
    hop = config.f("analysis.hop", 2.0)
    times = analysis_times(duration, hop)
    low_p = config.f("scoring.low_percentile", 10.0)
    high_p = config.f("scoring.high_percentile", 95.0)
    smooth = max(1, config.i("scoring.smoothing_frames", 2))

    normalized: dict[str, np.ndarray] = {}
    usable: dict[str, np.ndarray] = {}
    weight_map = {k: float(v) for k, v in config.section("weights").items() if isinstance(v, (int, float))}

    for name, signal in signals.items():
        if not signal.available:
            LOG.info("Señal %s desactivada: %s", name, signal.note or "no disponible")
            continue
        values = signal.resampled(times)
        smoothed = moving_average(values, smooth)
        norm = robust_normalize(smoothed, low_p, high_p)
        if float(norm.max()) < 1e-6:
            LOG.info("Señal %s ignorada: sin variación suficiente", name)
            continue
        normalized[name] = norm
        weight = weight_map.get(name, 0.0)
        if weight > 0:
            usable[name] = norm * weight

    if not usable:
        LOG.warning("No hay señales utilizables; no se pueden seleccionar clips.")
        return AnalysisResult(times, np.zeros(len(times), dtype=np.float32), signals, [], transcript)

    total_weight = sum(weight_map.get(name, 0.0) for name in usable)
    combined = np.sum(list(usable.values()), axis=0) / max(total_weight, 1e-6)
    combined = moving_average(combined, smooth)
    combined = robust_normalize(combined, 2.0, 99.0)
    combined = np.clip(combined, 0.0, 1.0).astype(np.float32)

    # Local maxima on the smoothed score.
    local: list[int] = [
        i for i in range(1, len(combined) - 1)
        if combined[i] >= combined[i - 1] and combined[i] > combined[i + 1]
    ]
    if not local:
        local = [int(np.argmax(combined))]
    top_score = float(np.max(combined))
    threshold = max(config.f("scoring.peak_threshold", 0.42), 0.02)
    candidates = [i for i in local if combined[i] >= threshold]
    wanted = config.i("selection.clips", 10)
    if len(candidates) < wanted * 2:
        relative = config.f("scoring.relative_threshold", 0.35)
        floor = max(0.02, top_score * relative)
        extra = [i for i in local if i not in candidates and combined[i] >= floor]
        candidates.extend(sorted(extra, key=lambda i: -combined[i])[: wanted * 3])

    # Merge peaks that are closer than merge_distance (keep the stronger one),
    # then rank by score.
    merge_distance = config.f("scoring.merge_distance", 8.0)
    candidates.sort(key=lambda i: times[i])
    merged: list[int] = []
    for idx in candidates:
        if merged and times[idx] - times[merged[-1]] <= merge_distance:
            if combined[idx] > combined[merged[-1]]:
                merged[-1] = idx
        else:
            merged.append(idx)
    merged.sort(key=lambda i: -combined[i])

    audio = signals.get("audio")
    envelope_signal = audio if audio is not None and audio.available else None
    min_peak_distance = config.f("scoring.min_peak_distance", 12.0)
    max_overlap = config.f("scoring.max_overlap", 0.0)

    accepted: list[tuple[float, float, float, int]] = []
    for idx in merged:
        if len(accepted) >= wanted:
            break
        peak = float(times[idx])
        peak_score = float(combined[idx])
        start, end = _clip_window(peak, duration, config)
        start, end = _snap_to_silence(start, end, peak, duration, envelope_signal, config)
        if any(abs(peak - p) < min_peak_distance for _, _, p, _ in accepted):
            continue
        overlap = False
        for s, e, _, _ in accepted:
            inter = max(0.0, min(end, e) - max(start, s))
            shorter = max(1e-6, min(end - start, e - s))
            if inter / shorter > max_overlap + 1e-6:
                overlap = True
                break
        if overlap:
            continue
        accepted.append((start, end, peak, idx))

    accepted.sort(key=lambda item: -combined[item[3]])
    clips: list[CandidateClip] = []
    for rank, (start, end, peak, idx) in enumerate(accepted, start=1):
        slc = _window_slice(times, start, end)
        clip_mean = float(np.mean(combined[slc])) if len(combined[slc]) else float(combined[idx])
        score = 0.7 * float(combined[idx]) + 0.3 * clip_mean
        reasons, breakdown = _reasons(
            start, end, times, normalized, signals, transcript, config
        )
        excerpt = transcript.excerpt(start, end) if transcript else ""
        title = _fallback_title(
            rank, start, end, transcript, signals.get("transcript")
        )
        clips.append(
            CandidateClip(
                index=rank,
                start=start,
                end=end,
                peak=peak,
                score=score,
                breakdown=breakdown,
                reasons=reasons,
                title=title,
                transcript=excerpt,
            )
        )

    # Rank by the final blended score, not only by the peak frame.
    clips.sort(key=lambda clip: -clip.score)
    for rank, clip in enumerate(clips, start=1):
        clip.index = rank
        if re.fullmatch(r"Momento destacado \d+", clip.title or ""):
            clip.title = f"Momento destacado {rank}"

    LOG.info(
        "Seleccionados %d clips de %d candidatos (umbral %.2f).",
        len(clips),
        len(merged),
        threshold,
    )
    return AnalysisResult(times, combined, signals, clips, transcript)


def candidate_snapshot(result: AnalysisResult) -> dict[str, Any]:
    """JSON-friendly summary of the analysis (used by --dry-run and tests)."""
    return {
        "frames": int(len(result.times)),
        "signals": {name: sig.to_meta_dict() for name, sig in result.signals.items()},
        "clips": [clip.to_dict() for clip in result.clips],
        "score_p95": float(np.percentile(result.scores, 95)) if len(result.scores) else 0.0,
    }
