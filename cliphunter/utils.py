"""Process, filesystem, formatting and logging helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rich.console import Console
from rich.logging import RichHandler

from .exceptions import ClipHunterError, DependencyError, SourceError
from .models import MediaInfo

console = Console(stderr=False, highlight=False)
LOG = logging.getLogger("cliphunter")

_FFMPEG_HINTS = {
    "win32": (
        "Instala ffmpeg con:  winget install Gyan.FFmpeg\n"
        "  (o: choco install ffmpeg). Cierra y vuelve a abrir la terminal."
    ),
    "darwin": "Instala ffmpeg con:  brew install ffmpeg",
    "linux": "Instala ffmpeg con:  sudo apt install ffmpeg   (o dnf/pacman según tu distro)",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure rich logging once for the whole application."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        root.addHandler(
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                show_path=verbose,
                markup=False,
            )
        )
    root.setLevel(level)
    LOG.setLevel(level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("yt_dlp").setLevel(logging.WARNING)
    logging.getLogger("chat_downloader").setLevel(logging.WARNING)


def ffmpeg_path() -> str:
    """Return the ffmpeg executable, honoring ``CLIPHUNTER_FFMPEG``."""
    candidate = os.environ.get("CLIPHUNTER_FFMPEG") or shutil.which("ffmpeg")
    if not candidate:
        raise DependencyError(
            "No se encontró ffmpeg.\n  " + _FFMPEG_HINTS.get(sys.platform, _FFMPEG_HINTS["linux"])
        )
    return candidate


def ffprobe_path() -> str:
    """Return the ffprobe executable next to ffmpeg when possible."""
    env = os.environ.get("CLIPHUNTER_FFPROBE")
    if env:
        return env
    found = shutil.which("ffprobe")
    if found:
        return found
    sibling = Path(ffmpeg_path()).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if sibling.exists():
        return str(sibling)
    raise DependencyError("No se encontró ffprobe (suele instalarse junto con ffmpeg).")


def ensure_media_tools() -> None:
    """Fail early with an actionable message when ffmpeg is missing."""
    ffmpeg_path()
    ffprobe_path()


def popen_kwargs() -> dict[str, Any]:
    """Hide console windows for child processes on Windows."""
    if os.name == "nt":
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


def terminate_process_tree(proc: subprocess.Popen[Any], timeout: float = 10.0) -> None:
    """Terminate a process and all of its children (important on Windows)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                **popen_kwargs(),
            )
        except OSError:
            proc.terminate()
    else:
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass


def run_capture(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capture text output and raise a clean error on failure."""
    LOG.debug("Running: %s", " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.run(
            [str(c) for c in cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
            **popen_kwargs(),
        )
    except FileNotFoundError as exc:
        raise DependencyError(f"Ejecutable no encontrado: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ClipHunterError(f"El comando tardó demasiado: {cmd[0]}") from exc

    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        tail = "\n".join(detail.splitlines()[-8:])
        raise ClipHunterError(f"Falló {Path(str(cmd[0])).name} (código {proc.returncode}):\n{tail}")
    return proc


def probe_media(path: str | Path) -> MediaInfo:
    """Read duration, resolution, fps and audio availability with ffprobe."""
    media = Path(path)
    if not media.is_file():
        raise SourceError(f"No existe el archivo: {media}")
    if media.stat().st_size == 0:
        raise SourceError(f"El archivo está vacío: {media}")

    cmd = [
        ffprobe_path(),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(media),
    ]
    proc = run_capture(cmd)
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise SourceError(f"ffprobe no pudo leer {media}") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise SourceError(f"El archivo no contiene video: {media}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise SourceError(f"No se pudo determinar la duración de {media}")

    fps = 0.0
    rate = str(video.get("r_frame_rate", "0/1"))
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return MediaInfo(
        path=media,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        has_audio=has_audio,
    )


def probe_duration(path: str | Path) -> float:
    """Cheap duration probe for already validated media."""
    return probe_media(path).duration


_TIME_PART_RE = re.compile(r"^\d+(?:\.\d+)?$")


def parse_timestamp(value: str) -> float:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS`` into seconds.

    Decimal seconds are accepted (``"1:30.5"``). Colon-separated minute and
    second parts must be lower than 60; the leading part may be any size.
    """
    raw = str(value).strip()
    if not raw:
        raise ClipHunterError("El tiempo no puede estar vacío.")

    parts = raw.split(":")
    if len(parts) > 3 or any(not _TIME_PART_RE.match(part) for part in parts):
        raise ClipHunterError(
            f"Formato de tiempo inválido: {value!r}. Usa SS, MM:SS o HH:MM:SS."
        )

    numbers = [float(part) for part in parts]
    if len(parts) >= 2 and numbers[-1] >= 60:
        raise ClipHunterError(
            f"Formato de tiempo inválido: {value!r}. Los segundos deben ser < 60."
        )
    if len(parts) == 3 and numbers[1] >= 60:
        raise ClipHunterError(
            f"Formato de tiempo inválido: {value!r}. Los minutos deben ser < 60."
        )

    if len(parts) == 3:
        hours, minutes, seconds = numbers
    elif len(parts) == 2:
        hours = 0.0
        minutes, seconds = numbers
    else:
        hours, minutes, seconds = 0.0, 0.0, numbers[0]
    return hours * 3600.0 + minutes * 60.0 + seconds


def format_hms(seconds: float) -> str:
    """Format seconds as ``H:MM:SS`` (or ``MM:SS`` under an hour)."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_ass_time(seconds: float) -> str:
    """Format seconds for an ASS timestamp (``H:MM:SS.cc``)."""
    total_cs = max(0, int(round(float(seconds) * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def slugify(text: str, max_len: int = 45) -> str:
    """Turn a title into a safe ASCII filename fragment."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return (slug[:max_len].rstrip("-")) or "clip"


def json_default(value: Any) -> Any:
    """JSON encoder for Path, numpy and set values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON atomically so interrupted runs never leave half files."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    tmp.replace(target)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text_atomic(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)


def moving_average(values: np.ndarray, span: int) -> np.ndarray:
    """Centered moving average that tolerates short inputs."""
    if span <= 1 or len(values) == 0:
        return values.astype(np.float32, copy=True)
    span = min(int(span), len(values))
    kernel = np.ones(span, dtype=np.float32) / span
    pad = span // 2
    padded = np.pad(values.astype(np.float32), (pad, span - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def robust_normalize(
    values: np.ndarray,
    low_percentile: float = 10.0,
    high_percentile: float = 95.0,
) -> np.ndarray:
    """Percentile-robust normalization into [0, 1]."""
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0 or float(arr.max()) - float(arr.min()) < 1e-9:
        return np.zeros_like(arr)
    low = float(np.percentile(arr, low_percentile))
    high = float(np.percentile(arr, high_percentile))
    if high - low < 1e-9:
        high = float(arr.max())
        low = float(arr.min())
    if high - low < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0).astype(np.float32)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def unique_paths(items: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for item in items:
        key = str(item.resolve()).lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
