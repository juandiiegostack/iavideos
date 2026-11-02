"""Audio extraction and energy / scream / laughter detection."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..cache import Cache
from ..config import Config
from ..exceptions import ClipHunterError
from ..models import MediaInfo, SignalSeries
from ..utils import LOG, ffmpeg_path, moving_average, probe_duration, robust_normalize, run_capture
from .base import aggregate_mean, analysis_times


def extract_audio(
    media: MediaInfo | str | Path,
    cache: Cache,
    key: str,
    config: Config,
    *,
    force: bool = False,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Decode the source to mono 16 kHz PCM WAV, cached under the source key.

    When ``start``/``end`` are provided only that section is decoded and the
    cache filename includes the range.
    """
    media_path = Path(media) if isinstance(media, (str, Path)) else Path(media.path)
    sample_rate = config.i("audio.sample_rate", 16000)
    partial = start is not None or end is not None
    if partial:
        tag = f"_r{int(round((start or 0.0) * 1000))}-{int(round((end or 0.0) * 1000))}"
    else:
        tag = ""
    target = cache.path(key, f"audio_{sample_rate}{tag}.wav")
    if target.is_file() and target.stat().st_size > 1024 and not force:
        LOG.debug("Audio en caché: %s", target)
        return target

    LOG.info(
        "Extrayendo audio%s (%d Hz mono)…",
        f" de {start:.1f}-{end:.1f}s" if partial else "",
        sample_rate,
    )
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, float(start)):.3f}"]
    cmd += ["-i", str(media_path.resolve())]
    if end is not None:
        cmd += ["-t", f"{max(0.05, float(end) - float(start or 0.0)):.3f}"]
    cmd += [
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        run_capture(cmd, timeout=3600)
    except Exception as exc:  # noqa: BLE001
        raise ClipHunterError(
            "No se pudo extraer el audio (¿el video no tiene pista de audio?).\n"
            f"Detalle: {exc}"
        ) from exc
    if not target.is_file() or target.stat().st_size <= 1024:
        raise ClipHunterError("ffmpeg no generó el audio del video.")
    return target


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if width != 2:
        raise ClipHunterError(f"Se esperaba audio PCM 16-bit, se obtuvo {width * 8}-bit.")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _modulation(smooth_db: np.ndarray, fine_times: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Detect 3-9 Hz amplitude modulation typical of laughter."""
    if len(fine_times) < 4:
        return np.zeros(len(fine_times), dtype=np.float32)
    rate = 20.0
    env_times = np.arange(fine_times[0], fine_times[-1] + 1e-6, 1.0 / rate)
    if len(env_times) < 16:
        return np.zeros(len(fine_times), dtype=np.float32)
    amplitude = np.power(10.0, smooth_db / 20.0).astype(np.float64)
    envelope = np.interp(env_times, fine_times, amplitude)
    envelope = envelope - moving_average(envelope, int(rate * 1.0))
    spectrum = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / rate)
    spectrum[(freqs < band[0]) | (freqs > band[1])] = 0
    rhythmic = np.fft.irfft(spectrum, len(envelope))
    span = max(2, int(rate * 4.0))
    energy = moving_average(rhythmic**2, span)
    total = moving_average(envelope**2, span) + 1e-10
    modulation = energy / total
    return np.interp(fine_times, env_times, modulation).astype(np.float32)


def analyze_audio(
    wav_path: str | Path,
    duration: float,
    config: Config,
    *,
    start: float = 0.0,
    end: float | None = None,
) -> SignalSeries:
    """Compute RMS, sudden-loudness, scream and laughter features.

    ``start``/``end`` select the analyzed section while all timestamps stay
    relative to the original video.
    """
    window = config.f("analysis.window", 10.0)
    hop = config.f("analysis.hop", 2.0)
    range_start = max(0.0, float(start))
    range_end = float(duration) if end is None else min(float(duration), float(end))
    span = max(0.0, range_end - range_start)
    times = analysis_times(span, hop, offset=range_start)
    sample_rate = config.i("audio.sample_rate", 16000)

    audio, rate = read_wav_mono(Path(wav_path))
    if rate != sample_rate:
        # The extraction step always uses this rate; resample defensively.
        target_len = int(len(audio) * sample_rate / rate)
        audio = np.interp(
            np.linspace(0, len(audio), target_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        rate = sample_rate
    if len(audio) == 0:
        return SignalSeries("audio", times, np.zeros(len(times), dtype=np.float32), False, "audio vacío")

    frame_len = max(256, int(round(rate * config.f("audio.frame_ms", 64.0) / 1000.0)))
    hop_len = max(64, int(round(rate * config.f("audio.hop_ms", 32.0) / 1000.0)))
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))

    view = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    num_frames = len(view)
    nfft = 1 << (frame_len - 1).bit_length()
    hann = np.hanning(frame_len).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
    scream_band = config.get("audio.scream_band_hz", [2500, 7500])
    scream_mask = (freqs >= float(scream_band[0])) & (freqs <= float(scream_band[1]))

    rms_db = np.empty(num_frames, dtype=np.float32)
    high_ratio = np.empty(num_frames, dtype=np.float32)
    chunk_size = 4096
    for start in range(0, num_frames, chunk_size):
        stop = min(num_frames, start + chunk_size)
        raw = view[start:stop]
        rms = np.sqrt(np.mean(raw * raw, axis=1) + 1e-12)
        rms_db[start:stop] = 20.0 * np.log10(rms + 1e-7)
        spectrum = np.fft.rfft(raw * hann, n=nfft, axis=1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        total = power.sum(axis=1) + 1e-12
        high_ratio[start:stop] = power[:, scream_mask].sum(axis=1) / total

    db_floor = config.f("audio.db_floor", -60.0)
    rms_db = np.clip(rms_db, db_floor, 3.0)
    fine_times = (
        (np.arange(num_frames, dtype=np.float64) * hop_len + frame_len / 2.0) / rate
        + range_start
    )
    smooth_db = moving_average(rms_db, max(1, int(round(0.3 * rate / hop_len))))
    local_mean = moving_average(smooth_db, max(1, int(round(3.0 * rate / hop_len))))
    novelty = np.maximum(0.0, smooth_db - local_mean)

    quiet_db = config.f("audio.quiet_db", -42.0)
    loudness = np.clip((smooth_db - quiet_db) / (3.0 - quiet_db), 0.0, 1.0)
    scream = high_ratio * loudness

    laughter_band = config.get("audio.laughter_band_hz", [3, 9])
    laughter = _modulation(
        smooth_db, fine_times, (float(laughter_band[0]), float(laughter_band[1]))
    )
    laughter = laughter * loudness

    rms_win = aggregate_mean(fine_times, smooth_db, times, window)
    novelty_win = aggregate_mean(fine_times, novelty, times, window)
    scream_win = aggregate_mean(fine_times, scream, times, window)
    laughter_win = aggregate_mean(fine_times, laughter, times, window)

    novelty_w = config.f("audio.novelty", 0.35)
    scream_w = config.f("audio.scream", 0.15)
    laughter_w = config.f("audio.laughter", 0.10)
    loud_w = max(0.0, 1.0 - novelty_w - scream_w - laughter_w)
    raw = (
        loud_w * robust_normalize(rms_win)
        + novelty_w * robust_normalize(novelty_win)
        + scream_w * robust_normalize(scream_win)
        + laughter_w * robust_normalize(laughter_win)
    )
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)

    # Coarse envelope used to snap clip cuts to quiet points.
    env_rate = 10
    env_end = max(range_end, float(fine_times[-1])) if len(fine_times) else range_end
    env_times = np.arange(range_start, env_end + 1e-6, 1.0 / env_rate)
    env_db = np.interp(env_times, fine_times, smooth_db, left=db_floor, right=db_floor)

    LOG.info(
        "Señal de audio: pico %.1f dB, mediana %.1f dB",
        float(np.percentile(rms_win, 99)),
        float(np.median(rms_win)),
    )
    return SignalSeries(
        "audio",
        times,
        raw,
        True,
        f"RMS mediana {np.median(rms_win):.1f} dB",
        detail={
            "rms_db": rms_win,
            "novelty": novelty_win,
            "scream": scream_win,
            "laughter": laughter_win,
        },
        envelope_times=env_times.astype(np.float32),
        envelope_db=env_db.astype(np.float32),
    )


def audio_duration(path: str | Path) -> float:
    """Duration helper for tests."""
    return probe_duration(path)
