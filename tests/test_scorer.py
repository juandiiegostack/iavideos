from __future__ import annotations

import numpy as np

from cliphunter.config import load_config
from cliphunter.models import (
    CandidateClip,
    SignalSeries,
    TranscriptData,
    TranscriptSegment,
    Word,
)
from cliphunter.scorer import _clip_window, _fallback_title, _snap_to_silence, score_and_select
from cliphunter.signals.base import analysis_times


def _series(name: str, duration: float, peaks: list[tuple[float, float]]) -> SignalSeries:
    times = analysis_times(duration, 2.0)
    values = np.zeros(len(times), dtype=np.float32)
    for center, height in peaks:
        index = int(np.argmin(np.abs(times - center)))
        values[index] = height
        # Give the peak a small plateau so smoothing keeps some signal.
        if index + 1 < len(values):
            values[index + 1] = height * 0.8
        if index > 0:
            values[index - 1] = height * 0.8
    return SignalSeries(name, times, values, True, "test")


def test_score_and_select_returns_non_overlapping_sorted_clips() -> None:
    duration = 180.0
    config = load_config(overrides={"selection.clips": 3, "selection.min_duration": 15, "selection.max_duration": 40})
    signals = {
        "chat": _series("chat", duration, [(30.0, 1.0), (100.0, 0.85)]),
        "audio": _series("audio", duration, [(60.0, 1.0), (150.0, 0.9)]),
    }
    result = score_and_select(signals, None, duration, config)

    assert 1 <= len(result.clips) <= 3
    assert len(result.times) == len(result.scores)
    assert all(15 <= clip.duration <= 40 for clip in result.clips)
    scores = [clip.score for clip in result.clips]
    assert scores == sorted(scores, reverse=True)
    for left, right in zip(result.clips, result.clips[1:]):
        overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
        shorter = min(left.duration, right.duration)
        assert overlap / shorter <= 1e-6


def test_score_and_select_without_signals_returns_empty() -> None:
    config = load_config()
    signals = {"chat": SignalSeries("chat", np.zeros(5), np.zeros(5), False, "n/a")}
    result = score_and_select(signals, None, 60.0, config)
    assert result.clips == []
    assert result.scores.max() == 0.0


def test_clip_window_clamps_to_media_bounds() -> None:
    config = load_config(
        overrides={
            "selection.min_duration": 15,
            "selection.max_duration": 30,
            "selection.pre_roll": 8,
            "selection.post_roll": 4,
        }
    )
    assert _clip_window(2.0, 200.0, config) == (0.0, 15.0)
    start, end = _clip_window(199.0, 200.0, config)
    assert end == 200.0
    assert abs((end - start) - 15.0) < 1e-6

    start, end = _clip_window(100.0, 200.0, config)
    assert abs((end - start) - 15.0) < 1e-6
    assert start <= 100.0 <= end


def test_clip_window_respects_max_duration() -> None:
    config = load_config(
        overrides={
            "selection.min_duration": 10,
            "selection.max_duration": 20,
            "selection.pre_roll": 40,
            "selection.post_roll": 40,
        }
    )
    start, end = _clip_window(100.0, 300.0, config)
    assert abs((end - start) - 20.0) < 1e-6


def test_clip_window_keeps_short_videos_whole() -> None:
    config = load_config()
    assert _clip_window(5.0, 12.0, config) == (0.0, 12.0)


def test_snap_to_silence_moves_boundary_to_quiet_point() -> None:
    config = load_config(overrides={"selection.silence_snap": 3.0, "audio.quiet_db": -42})
    env_t = np.arange(0.0, 61.0, 0.5, dtype=np.float32)
    env_db = np.full(len(env_t), -20.0, dtype=np.float32)
    env_db[int(8.0 / 0.5)] = -60.0
    env_db[int(44.0 / 0.5)] = -60.0
    signal = SignalSeries(
        "audio",
        env_t,
        np.zeros(len(env_t), dtype=np.float32),
        True,
        "test",
        envelope_times=env_t,
        envelope_db=env_db,
    )
    start, end = _snap_to_silence(10.0, 40.0, 25.0, 60.0, signal, config)
    assert start == 8.0
    assert end == 40.0


def test_snap_to_silence_without_envelope_is_noop() -> None:
    config = load_config()
    signal = SignalSeries("audio", np.zeros(3), np.zeros(3), True, "x")
    assert _snap_to_silence(1.0, 20.0, 10.0, 60.0, signal, config) == (1.0, 20.0)


def test_fallback_title_uses_keyword_and_strips_punctuation() -> None:
    transcript_signal = _series("transcript", 60.0, [(10.0, 1.0)])
    transcript_signal.detail["keyword_hits"] = [(10.0, "no puede ser", 1.25)]
    title = _fallback_title(1, 5.0, 20.0, None, transcript_signal)
    assert title == "No puede ser"

    transcript = TranscriptData(
        words=[Word(9.0, 9.5, "no"), Word(9.5, 10.0, "puede"), Word(10.0, 10.5, "ser")],
        segments=[TranscriptSegment(9.0, 11.0, "no puede ser,")],
    )
    assert _fallback_title(2, 5.0, 20.0, transcript, None) == "No puede ser"

    assert _fallback_title(4, 5.0, 20.0, None, None) == "Momento destacado 4"


def test_candidate_clip_duration_property() -> None:
    clip = CandidateClip(index=1, start=10.0, end=25.5, peak=12.0, score=0.5)
    assert clip.duration == 15.5
    assert clip.to_dict()["duration"] == 15.5
