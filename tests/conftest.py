"""Shared pytest fixtures (all media is generated locally, no network)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe no están instalados")
    return True


@pytest.fixture
def tiny_video(tmp_path: Path, has_ffmpeg: bool) -> Path:
    """A 20 s 640x360 test video with a sine audio track."""
    path = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x360:rate=25:duration=20",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=20",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def chat_messages() -> list[dict]:
    """Chat with a clear burst at t=10 s and quiet chatter elsewhere."""
    messages = []
    t = 0.5
    while t < 20:
        if 9.0 <= t <= 12.5:
            for _ in range(8):
                messages.append({"t": round(t, 2), "text": "KEKW no way insane", "emotes": []})
                t += 0.12
        else:
            messages.append({"t": round(t, 2), "text": "hola", "emotes": []})
            t += 1.5
    return messages


@pytest.fixture
def hype_video(tmp_path: Path, has_ffmpeg: bool) -> Path:
    """40 s video with two loud audio bursts (around 10 s and 30 s)."""
    import wave

    import numpy as np

    sample_rate = 16000
    duration = 40.0
    rng = np.random.default_rng(11)
    samples = int(sample_rate * duration)
    audio = rng.normal(0, 0.002, samples).astype(np.float32)
    for center in (10.0, 30.0):
        begin = int((center - 2.0) * sample_rate)
        count = int(4.0 * sample_rate)
        t = np.arange(count) / sample_rate
        burst = (np.sin(2 * np.pi * 2200 * t) * np.hanning(count) * 0.6).astype(np.float32)
        audio[begin : begin + count] += burst
    audio = np.clip(audio, -0.99, 0.99)

    wav_path = tmp_path / "hype_audio.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())

    video = tmp_path / "hype.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=640x360:rate=25:duration={int(duration)}",
            "-i",
            str(wav_path),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    return video

