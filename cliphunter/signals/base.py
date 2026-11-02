"""Shared helpers for sliding-window signal extraction."""

from __future__ import annotations

import numpy as np


def analysis_times(duration: float, hop: float, offset: float = 0.0) -> np.ndarray:
    """Window centers covering ``[offset, offset + duration]`` with a fixed step.

    ``offset`` keeps timestamps relative to the original source when only a
    section of a video is analyzed.
    """
    if duration <= 0 or hop <= 0:
        return np.zeros(0, dtype=np.float32)
    count = int(np.floor(duration / hop)) + 1
    times = np.arange(count, dtype=np.float64) * hop + offset
    if times[-1] < offset + duration - 1e-6:
        times = np.append(times, offset + duration)
    return times.astype(np.float32)


def counts_in_windows(
    event_times: np.ndarray,
    grid: np.ndarray,
    window: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Count (or weighted-sum) events inside a centered window per grid point."""
    if len(event_times) == 0:
        return np.zeros(len(grid), dtype=np.float32)
    order = np.argsort(event_times)
    times = np.asarray(event_times, dtype=np.float64)[order]
    w = (
        np.ones(len(times), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)[order]
    )
    cumulative = np.concatenate([[0.0], np.cumsum(w)])
    left = np.searchsorted(times, grid - window / 2.0, side="left")
    right = np.searchsorted(times, grid + window / 2.0, side="right")
    return (cumulative[right] - cumulative[left]).astype(np.float32)


def aggregate_mean(
    fine_times: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    window: float,
) -> np.ndarray:
    """Mean of a uniformly sampled series inside each analysis window."""
    if len(fine_times) == 0 or len(values) == 0:
        return np.zeros(len(grid), dtype=np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    left = np.searchsorted(fine_times, grid - window / 2.0, side="left")
    right = np.searchsorted(fine_times, grid + window / 2.0, side="right")
    sums = cumulative[right] - cumulative[left]
    counts = right - left
    out = np.zeros(len(grid), dtype=np.float32)
    nonempty = counts > 0
    out[nonempty] = (sums[nonempty] / counts[nonempty]).astype(np.float32)
    if np.any(~nonempty):
        out[~nonempty] = np.interp(grid[~nonempty], fine_times, values).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
