"""Clip cutting and encoding with ffmpeg (horizontal, vertical, subtitles)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import Config
from .models import CandidateClip, MediaInfo
from .utils import LOG, ffmpeg_path, run_capture, slugify

VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920


def _filter_escape(path: str) -> str:
    """Escape a filename for use inside an ffmpeg filter argument."""
    # We normally pass a relative filename with no spaces, but be defensive.
    return (
        path.replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace(" ", r"\ ")
    )


def build_filters(
    *,
    vertical: bool,
    vertical_mode: str,
    blur: int,
    brightness: float,
    subtitle_name: str | None,
    subtitle_resolution: tuple[int, int] | None,
) -> tuple[list[str], str | None]:
    """Return (``-filter_complex`` parts, final video label)."""
    parts: list[str] = []
    if vertical:
        if vertical_mode == "crop":
            parts.append(
                f"[0:v]scale={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={VERTICAL_WIDTH}:{VERTICAL_HEIGHT},setsar=1[vbase]"
            )
        else:
            parts.append(
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={VERTICAL_WIDTH}:{VERTICAL_HEIGHT},boxblur={max(2, blur)}:2,"
                f"eq=brightness={brightness:.3f}[bgblur];"
                f"[fg]scale={VERTICAL_WIDTH}:{VERTICAL_HEIGHT}:force_original_aspect_ratio=decrease,"
                f"setsar=1[fgscaled];"
                f"[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2[vbase]"
            )
        label = "vbase"
    else:
        if subtitle_name:
            parts.append("[0:v]setsar=1[vbase]")
            label = "vbase"
        else:
            label = ""

    if subtitle_name:
        res = subtitle_resolution or (
            (VERTICAL_WIDTH, VERTICAL_HEIGHT) if vertical else (1920, 1080)
        )
        # libass scales the ASS canvas to the actual frame; PlayRes in the file
        # already matches the output, so no extra scaling is required.
        parts.append(f"[{label}]ass={_filter_escape(subtitle_name)}[vout]")
        label = "vout"

    return parts, label or None


def cut_segment(
    media: MediaInfo | str | Path,
    start: float,
    end: float,
    output: str | Path,
    *,
    config: Config,
    vertical: bool = False,
    subtitle_path: str | Path | None = None,
    title: str = "",
    log: bool = True,
) -> Path:
    """Cut ``[start, end]`` with a frame-accurate re-encode.

    Vertical mode composes the video over a blurred, cropped background (or does
    a centered crop). Subtitles are burned from an ASS file.
    """
    media_path = Path(media) if isinstance(media, (str, Path)) else Path(media.path)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(start))
    duration = max(0.05, float(end) - start)

    subtitle_name: str | None = None
    if subtitle_path:
        sub = Path(subtitle_path)
        if not sub.is_file():
            raise FileNotFoundError(f"No existe el subtítulo: {sub}")
        sub_abs = sub.resolve()
        # Run ffmpeg from the clip folder and pass a bare filename: this avoids
        # Windows drive-letter escaping problems in filter graphs.
        out_dir = out.resolve().parent
        try:
            subtitle_name = sub_abs.relative_to(out_dir).as_posix()
        except ValueError:
            subtitle_name = sub_abs.as_posix()
        if sub.resolve().parent != out_dir:
            raise ValueError("El archivo ASS debe estar en la misma carpeta que el clip")

    filters, label = build_filters(
        vertical=vertical,
        vertical_mode=str(config.get("export.vertical_mode", "blur")),
        blur=config.i("export.vertical_blur", 28),
        brightness=config.f("export.vertical_brightness", -0.06),
        subtitle_name=subtitle_name,
        subtitle_resolution=(VERTICAL_WIDTH, VERTICAL_HEIGHT) if vertical else None,
    )

    cmd: list[str] = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(media_path.resolve()),
        "-t",
        f"{duration:.3f}",
    ]
    if filters:
        cmd += ["-filter_complex", ";".join(filters), "-map", f"[{label}]"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-map", "0:a:0?"]
    cmd += [
        "-c:v",
        str(config.get("export.codec", "libx264")),
        "-preset",
        str(config.get("export.preset", "veryfast")),
        "-crf",
        str(config.i("export.crf", 20)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        str(config.get("export.audio_bitrate", "192k")),
        "-movflags",
        "+faststart",
    ]
    if title:
        cmd += ["-metadata", f"title={title}"]
    cmd += [str(out.resolve())]

    if log:
        LOG.info("Cortando %s (%s - %s)…", out.name, f"{start:.1f}s", f"{end:.1f}s")
    try:
        run_capture(cmd, cwd=out.resolve().parent if subtitle_name else None)
    except subprocess.SubprocessError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"ffmpeg falló al cortar {out.name}: {exc}") from exc

    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg no generó el clip: {out}")
    return out


def clip_filename(clip: CandidateClip, out_dir: Path, index: int | None = None) -> Path:
    """Build a stable, readable ``clip_001_title-slug.mp4`` filename.

    When the target already exists a numeric suffix is added
    (``clip_001_title-slug_2.mp4``) so previous clips are never overwritten.
    """
    number = clip.index if index is None else index
    slug = slugify(clip.title or f"momento {number}", 40)
    target = out_dir / f"clip_{number:03d}_{slug}.mp4"
    if not target.exists():
        return target
    stem = target.stem
    suffix = 2
    while True:
        candidate = out_dir / f"{stem}_{suffix}{target.suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def export_clip(
    clip: CandidateClip,
    media: MediaInfo,
    out_dir: Path,
    config: Config,
    *,
    vertical: bool,
    subtitle_path: str | Path | None,
    overwrite: bool = True,
) -> Path:
    """Export one selected clip and annotate ``clip`` with its file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = clip_filename(clip, out_dir)
    if target.exists() and not overwrite:
        LOG.info("Clip ya existe, se omite: %s", target.name)
        clip.file = target.name
        return target
    cut_segment(
        media,
        clip.start,
        clip.end,
        target,
        config=config,
        vertical=vertical,
        subtitle_path=subtitle_path,
        title=clip.title,
    )
    clip.file = target.name
    if subtitle_path:
        clip.subtitle_file = Path(subtitle_path).name
    return target


def ffmpeg_supports_filter(name: str) -> bool:
    """Check whether the installed ffmpeg has a filter (ass/libass, etc.)."""
    try:
        proc = run_capture([ffmpeg_path(), "-hide_banner", "-filters"], check=False)
    except Exception:  # noqa: BLE001
        return False
    return any(f" {name} " in line for line in proc.stdout.splitlines())
