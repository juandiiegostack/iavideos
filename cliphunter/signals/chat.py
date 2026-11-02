"""Chat activity, emote density and hype-word signal."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..config import Config
from ..models import ChatMessage, SignalSeries
from ..utils import LOG, moving_average, robust_normalize
from .base import analysis_times, counts_in_windows


def _compile_patterns(config: Config) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
    emotes = [str(e) for e in config.get("chat.emotes", []) if str(e).strip()]
    hype = [str(w).strip().lower() for w in config.get("chat.hype_words", []) if str(w).strip()]
    emote_pattern = (
        re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(e) for e in sorted(emotes, key=len, reverse=True)) + r")(?!\w)"
        )
        if emotes
        else None
    )
    hype_pattern = (
        re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(w) for w in sorted(hype, key=len, reverse=True)) + r")(?!\w)",
            re.IGNORECASE,
        )
        if hype
        else None
    )
    return emote_pattern, hype_pattern


def analyze_chat(
    messages: list[ChatMessage] | None,
    duration: float,
    config: Config,
) -> SignalSeries:
    """Build the chat signal on the shared analysis grid."""
    window = config.f("analysis.window", 10.0)
    hop = config.f("analysis.hop", 2.0)
    times = analysis_times(duration, hop)

    if not messages:
        return SignalSeries(
            "chat", times, np.zeros(len(times), dtype=np.float32), False,
            "chat no disponible",
        )

    min_messages = config.i("chat.min_messages", 20)
    usable = [m for m in messages if 0 <= m.t <= duration + window and m.text]
    if len(usable) < min_messages:
        return SignalSeries(
            "chat", times, np.zeros(len(times), dtype=np.float32), False,
            f"solo {len(usable)} mensajes de chat",
        )

    emote_pattern, hype_pattern = _compile_patterns(config)
    msg_times = np.array([m.t for m in usable], dtype=np.float64)
    emote_counts = np.zeros(len(usable), dtype=np.float64)
    hype_counts = np.zeros(len(usable), dtype=np.float64)
    for i, message in enumerate(usable):
        text = message.text
        if message.emotes:
            emote_counts[i] = len(message.emotes)
        elif emote_pattern is not None:
            emote_counts[i] = len(emote_pattern.findall(text))
        if hype_pattern is not None:
            hype_counts[i] = len(hype_pattern.findall(text))

    counts = counts_in_windows(msg_times, times, window)
    emote_sum = counts_in_windows(msg_times, times, window, emote_counts)
    hype_sum = counts_in_windows(msg_times, times, window, hype_counts)

    rate = counts / window
    baseline_span = max(1, int(round(config.f("chat.moving_average_seconds", 90.0) / hop)))
    baseline = moving_average(rate, baseline_span)
    denom = np.maximum(baseline, max(0.02, float(np.median(rate)) * 0.1))
    spike = np.maximum(0.0, rate - baseline) / (denom + 1e-6)

    emote_density = emote_sum / np.maximum(counts, 1.0)
    emote_baseline = moving_average(emote_density, baseline_span)
    emote_excess = np.maximum(0.0, emote_density - emote_baseline)
    hype_density = hype_sum / np.maximum(counts, 1.0)

    raw = (
        0.45 * robust_normalize(spike)
        + 0.25 * robust_normalize(rate)
        + 0.20 * robust_normalize(emote_excess)
        + 0.10 * robust_normalize(hype_density)
    )
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)

    top_emotes = Counter()
    for message in usable:
        for emote in message.emotes:
            top_emotes[emote] += 1
    note = f"{len(usable)} mensajes"
    if top_emotes:
        note += "; emotes: " + ", ".join(name for name, _ in top_emotes.most_common(3))
    LOG.info("Señal de chat: %s", note)

    return SignalSeries(
        "chat",
        times,
        raw,
        True,
        note,
        detail={
            "rate": rate,
            "spike": spike,
            "emote_density": emote_density,
            "emote_excess": emote_excess,
            "hype_density": hype_density,
            "messages": np.array([len(usable)], dtype=np.float32),
        },
    )
