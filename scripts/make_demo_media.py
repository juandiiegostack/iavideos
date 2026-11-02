#!/usr/bin/env python
"""Generate a synthetic demo stream (video + chat) without downloading anything.

The demo contains three clearly "hype" moments so you can try scoring, vertical
export and the rest of the pipeline offline:

    python scripts/make_demo_media.py
    python cliphunter.py --file demo/demo_stream.mp4 --chat-file demo/demo_chat.json \
        --clips 3 --vertical --no-transcript

Whisper is disabled in the suggested command because the synthetic audio has no
real speech; you can still run with --subtitles to verify the graceful fallback
when there are no transcribed words.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cliphunter.utils import ffmpeg_path  # noqa: E402

EMOTES = ["KEKW", "LUL", "OMEGALUL", "PogChamp", "POGGERS", "monkaS", "catJAM", "HYPERS"]
HYPE = [
    "no way",
    "insane",
    "clutch",
    "lets go",
    "gg",
    "ez",
    "clip it",
    "increible",
    "que locura",
    "jajaja",
    "wtf",
    "omg",
    "pog",
]
CHILL = ["hola", "gg", "nice", "buena", "vamos", "juego", "eso", "jajaja"]


def _burst_audio(rng: np.random.Generator, sample_rate: int, seconds: float) -> np.ndarray:
    """Build a loud, modulated burst that looks like a reaction + crowd."""
    count = int(seconds * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.hanning(count).astype(np.float32) ** 0.35

    # Shout: sweeping fundamental with a few harmonics.
    sweep = 210.0 + 120.0 * np.sin(2 * math.pi * 0.35 * t)
    phase = 2 * math.pi * np.cumsum(sweep) / sample_rate
    shout = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)

    # Scream band noise.
    noise = rng.normal(0, 1, count).astype(np.float32)
    noise = np.diff(noise, prepend=0.0)
    scream = np.convolve(noise, np.ones(8, dtype=np.float32) / 8, mode="same")

    # Laughter-like 5.5 Hz amplitude modulation.
    laugh = (0.55 + 0.45 * np.sin(2 * math.pi * 5.5 * t)).astype(np.float32)

    mixed = (0.55 * shout + 0.45 * scream * laugh) * envelope
    return (mixed * 0.55).astype(np.float32)


def make_audio(
    duration: float, sample_rate: int, burst_centers: list[float], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = int(duration * sample_rate)
    audio = rng.normal(0, 0.004, total).astype(np.float32)
    t = np.arange(total, dtype=np.float32) / sample_rate
    audio += 0.002 * np.sin(2 * math.pi * 120 * t)

    for center in burst_centers:
        burst = _burst_audio(rng, sample_rate, 7.0)
        begin = int(max(0.0, center - 3.0) * sample_rate)
        stop = min(total, begin + len(burst))
        audio[begin:stop] += burst[: stop - begin]

    # Occasional small isolated sounds so the score has some texture.
    for center in burst_centers:
        offset = int((center + 14.0) * sample_rate)
        if offset + sample_rate < total:
            t2 = np.arange(sample_rate, dtype=np.float32) / sample_rate
            blip = 0.05 * np.sin(2 * math.pi * 900 * t2) * np.hanning(sample_rate)
            audio[offset : offset + sample_rate] += blip.astype(np.float32)

    audio = np.tanh(audio * 1.4)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.89).astype(np.float32)


def make_chat(duration: float, burst_centers: list[float], seed: int) -> list[dict]:
    rng = random.Random(seed)
    messages: list[dict] = []

    def in_burst(t: float) -> bool:
        return any(center - 3.0 <= t <= center + 4.0 for center in burst_centers)

    t = 1.0
    while t < duration - 1.0:
        if in_burst(t):
            for _ in range(rng.randint(3, 6)):
                words = [rng.choice(HYPE) for _ in range(rng.randint(1, 2))]
                emotes = [rng.choice(EMOTES) for _ in range(rng.randint(0, 3))]
                messages.append(
                    {
                        "t": round(t + rng.uniform(0.0, 0.4), 2),
                        "text": " ".join(words + emotes).strip(),
                        "emotes": [],
                    }
                )
                t += 0.18
        else:
            if rng.random() < 0.5:
                messages.append(
                    {
                        "t": round(t, 2),
                        "text": rng.choice(CHILL),
                        "emotes": [],
                    }
                )
            t += rng.uniform(1.5, 3.5)

    messages.sort(key=lambda item: item["t"])
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="demo", help="Output directory (default: demo).")
    parser.add_argument("--duration", type=float, default=120.0, help="Duration in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    args = parser.parse_args(argv)

    if args.duration < 30:
        parser.error("--duration debe ser >= 30 segundos.")

    try:
        ffmpeg = ffmpeg_path()
    except Exception as exc:  # noqa: BLE001 - friendly message
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    video = out / "demo_stream.mp4"
    chat_file = out / "demo_chat.json"
    wav_file = out / "_demo_audio.wav"

    burst_centers = [round(args.duration * (index + 0.5) / 3.0, 1) for index in range(3)]
    sample_rate = 48000
    audio = make_audio(args.duration, sample_rate, burst_centers, args.seed)
    with wave.open(str(wav_file), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=960x540:rate=30:duration={args.duration:g}",
        "-i",
        str(wav_file),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        str(video),
    ]
    print(f"Generando video sintetico ({args.duration:.0f}s) con ffmpeg...")
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    wav_file.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = "\n".join((result.stderr or "").strip().splitlines()[-6:])
        print(f"ERROR: ffmpeg fallo al generar el video:\n{detail}", file=sys.stderr)
        return result.returncode

    chat = make_chat(args.duration, burst_centers, args.seed)
    chat_file.write_text(json.dumps(chat, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Video: {video}")
    print(f"Chat:  {chat_file} ({len(chat)} mensajes)")
    print("Momentos calientes inyectados: " + ", ".join(f"{t:.0f}s" for t in burst_centers))
    print("\nPrueba rapida:")
    print(
        f'  python cliphunter.py --file "{video}" --chat-file "{chat_file}" '
        "--clips 3 --vertical --no-transcript --out demo/clips"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
