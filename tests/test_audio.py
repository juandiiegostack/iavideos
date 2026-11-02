from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cliphunter.config import load_config
from cliphunter.signals.audio import analyze_audio, read_wav_mono


def _write_test_wav(path: Path, duration: float = 40.0, sample_rate: int = 16000) -> Path:
    rng = np.random.default_rng(3)
    samples = int(duration * sample_rate)
    audio = rng.normal(0, 0.008, samples).astype(np.float32)
    for start, length, frequency, gain in [
        (11.75, 0.5, 3000, 0.8),
        (29.75, 0.5, 4500, 0.7),
    ]:
        begin = int(start * sample_rate)
        count = int(length * sample_rate)
        t = np.arange(count) / sample_rate
        burst = (np.sin(2 * np.pi * frequency * t) * np.hanning(count) * gain).astype(np.float32)
        audio[begin : begin + count] += burst
    audio = np.clip(audio, -0.99, 0.99)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
    return path


def test_audio_peaks_align_with_loud_bursts(tmp_path: Path) -> None:
    path = _write_test_wav(tmp_path / "audio.wav")
    config = load_config()
    signal = analyze_audio(path, 40.0, config)
    assert signal.available
    assert signal.envelope_times is not None
    assert signal.envelope_db is not None
    assert len(signal.envelope_times) == len(signal.envelope_db)
    peak_time = float(signal.times[int(signal.values.argmax())])
    assert min(abs(peak_time - 12.0), abs(peak_time - 30.0)) <= 2.0
    assert float(signal.values.max()) > 0.7
    assert set(signal.detail) == {"rms_db", "novelty", "scream", "laughter"}


def test_read_wav_mono_handles_stereo(tmp_path: Path, has_ffmpeg: bool) -> None:
    # Generate a stereo wav through numpy (no ffmpeg required).
    path = tmp_path / "stereo.wav"
    left = np.ones(1600, dtype=np.float32) * 0.5
    right = np.ones(1600, dtype=np.float32) * -0.5
    interleaved = np.empty(3200, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((interleaved * 32767).astype(np.int16).tobytes())
    audio, rate = read_wav_mono(path)
    assert rate == 16000
    assert len(audio) == 1600
    assert np.max(np.abs(audio)) < 0.01
