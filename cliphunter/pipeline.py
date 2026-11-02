"""End-to-end orchestration: acquire, analyze, score, export."""

from __future__ import annotations

import hashlib
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from . import __version__
from .cache import Cache, source_key
from .config import Config
from .downloader import (
    LiveRecorder,
    SourceSpec,
    classify_url,
    download_vod,
    fetch_chat,
    fetch_remote_info,
    local_source,
)
from .exceptions import (
    ChatUnavailable,
    ClipHunterError,
    LLMError,
    SourceError,
    TranscriptionError,
)
from .exporter import clip_filename, export_clip
from .llm import LLMClient
from .models import AnalysisResult, CandidateClip, MediaInfo, SignalSeries, TranscriptData
from .scorer import candidate_snapshot, score_and_select
from .signals.audio import analyze_audio, extract_audio
from .signals.chat import analyze_chat
from .signals.query import analyze_query
from .signals.transcript import (
    analyze_transcript,
    load_cached_transcript,
    transcribe,
    transcript_cache_name,
)
from .subtitles import build_ass
from .utils import (
    LOG,
    atomic_write_json,
    console,
    ensure_media_tools,
    format_hms,
    probe_media,
)


@dataclass
class RunOptions:
    """Non-config runtime options."""

    url: str | None = None
    file: str | None = None
    live: bool = False
    max_duration: float = 5.0
    start: float | None = None
    end: float | None = None
    query: str | None = None
    chat_file: str | None = None
    cookies_from_browser: str | None = None
    cookie_file: str | None = None
    no_cache: bool = False
    clear_cache: bool = False
    force: bool = False
    dry_run: bool = False


@dataclass
class PipelineResult:
    media: MediaInfo
    clips: list[CandidateClip]
    output_dir: Path
    manifest_path: Path | None
    cache_key: str


class _TaskProgress:
    def __init__(self, progress: Progress, task_id: Any) -> None:
        self.progress = progress
        self.task_id = task_id

    def __call__(self, fraction: float, message: str) -> None:
        self.progress.update(
            self.task_id,
            completed=max(0.0, min(100.0, fraction * 100.0)),
            description=message,
        )


@contextmanager
def _progress_ui() -> Iterator[Progress]:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with progress:
        yield progress


def _signal_payload(signal: SignalSeries) -> dict[str, Any]:
    return {
        "name": signal.name,
        "times": signal.times,
        "values": signal.values,
        "available": signal.available,
        "note": signal.note,
        "detail": signal.detail,
        "envelope_times": signal.envelope_times,
        "envelope_db": signal.envelope_db,
    }


def _maybe_array(value: Any) -> Any:
    if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
        return np.asarray(value, dtype=np.float32)
    return value


def _load_signals_cache(
    cache: Cache, key: str, filename: str, duration: float
) -> dict[str, SignalSeries] | None:
    data = cache.load_json(key, filename)
    if not isinstance(data, dict):
        return None
    if abs(float(data.get("duration", -1)) - duration) > 1.0:
        return None
    signals: dict[str, SignalSeries] = {}
    for name, payload in (data.get("signals") or {}).items():
        try:
            times = np.asarray(payload["times"], dtype=np.float32)
            values = np.asarray(payload["values"], dtype=np.float32)
            if len(times) != len(values) or len(times) == 0:
                return None
            detail = {k: _maybe_array(v) for k, v in (payload.get("detail") or {}).items()}
            env_t = payload.get("envelope_times")
            env_db = payload.get("envelope_db")
            signals[name] = SignalSeries(
                name=payload.get("name", name),
                times=times,
                values=values,
                available=bool(payload.get("available", True)),
                note=str(payload.get("note", "")),
                detail=detail,
                envelope_times=np.asarray(env_t, dtype=np.float32) if env_t else None,
                envelope_db=np.asarray(env_db, dtype=np.float32) if env_db else None,
            )
        except (KeyError, TypeError, ValueError):
            LOG.warning("Caché de señales corrupta; se recalcula.")
            return None
    return signals if signals else None


def _save_signals_cache(
    cache: Cache,
    key: str,
    filename: str,
    signals: dict[str, SignalSeries],
    duration: float,
) -> None:
    payload = {
        "duration": duration,
        "signals": {name: _signal_payload(sig) for name, sig in signals.items()},
    }
    cache.save_json(key, filename, payload)


def _resolve_analysis_range(options: RunOptions, duration: float) -> tuple[float, float]:
    """Validate --start/--end against the real media duration."""
    start = max(0.0, float(options.start or 0.0))
    end = float(duration) if options.end is None else float(options.end)
    if start >= float(duration) - 1e-6:
        raise ClipHunterError(
            f"--start ({format_hms(start)}) está fuera del video "
            f"(duración {format_hms(duration)})."
        )
    if end <= start + 1e-6:
        raise ClipHunterError("--start debe ser menor que --end.")
    if end > float(duration) + 1e-6:
        LOG.warning(
            "--end (%s) supera la duración del video (%s); se usará el final.",
            format_hms(end),
            format_hms(duration),
        )
        end = float(duration)
    return start, end


def _resolve_source(
    options: RunOptions,
    config: Config,
    cache: Cache,
    progress: Progress,
) -> tuple[MediaInfo, str, SourceSpec | None]:
    """Resolve local file, VOD download or live recording."""
    if options.file:
        if options.live:
            raise SourceError("--live no se puede combinar con --file.")
        media = local_source(options.file)
        media.title = media.title or Path(media.path).stem
        media.platform = "local"
        key = source_key(local_path=media.path)
        if options.clear_cache:
            cache.clear(key)
        return media, key, None

    if not options.url:
        raise SourceError("Indica una URL o usa --file ruta.mp4.")

    spec = classify_url(options.url)
    if options.live:
        # A YouTube live stream uses the same /watch?v= shape as a VOD, so for
        # those URLs we must ask yt-dlp whether it is actually live. Direct
        # .m3u8 URLs skip this check (yt-dlp may not know how to probe them).
        if spec.kind in {"vod", "live"}:
            task = progress.add_task("Comprobando directo", total=100)
            remote = fetch_remote_info(
                spec,
                cookies_from_browser=options.cookies_from_browser,
                cookie_file=options.cookie_file,
            )
            progress.update(task, completed=100, description="Directo comprobado")
            if not remote.is_live:
                raise SourceError(
                    "Esa URL no es un directo en vivo. Quita --live para descargarla "
                    "como VOD (o usa la URL del canal cuando esté emitiendo)."
                )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        key = f"live_{spec.platform}_{spec.channel or spec.video_id or 'stream'}_{stamp}"
        task = progress.add_task("Grabando en directo", total=100)
        recorder = LiveRecorder(cache, key, config)
        previous_handler = signal.getsignal(signal.SIGINT)

        def handle_interrupt(signum: int, frame: Any) -> None:
            if recorder._stop.is_set():  # second Ctrl+C: abort immediately
                raise KeyboardInterrupt
            LOG.warning(
                "Ctrl+C: se detendrá la grabación y se analizará lo grabado. "
                "Pulsa Ctrl+C otra vez para cancelar del todo."
            )
            recorder.request_stop()

        signal.signal(signal.SIGINT, handle_interrupt)
        try:
            media = recorder.record(
                options.url,
                options.max_duration,
                on_progress=_TaskProgress(progress, task),
                record_chat=config.b("analysis.signals.chat", True),
            )
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        media.source_url = options.url
        media.video_id = spec.channel or spec.video_id
        if spec.platform == "direct":
            media.video_id = Path(options.url.split("?")[0]).stem or "stream"
        media.platform = spec.platform
        media.title = f"Directo de {media.video_id or spec.platform}"
        return media, key, spec

    task = progress.add_task("Leyendo información del video", total=100)
    remote = fetch_remote_info(
        spec,
        cookies_from_browser=options.cookies_from_browser,
        cookie_file=options.cookie_file,
    )
    progress.update(task, completed=100, description="Información obtenida")
    if remote.is_live:
        raise SourceError(
            "Ese enlace es un directo en vivo. Añade --live y --max-duration N "
            "(minutos) para grabar y analizar."
        )
    key = source_key(platform=remote.platform, video_id=remote.video_id, url=remote.url)
    if options.clear_cache:
        cache.clear(key)
    task = progress.add_task("Descargando video", total=100)
    media = download_vod(
        remote,
        cache,
        key,
        cookies_from_browser=options.cookies_from_browser,
        cookie_file=options.cookie_file,
        on_progress=_TaskProgress(progress, task),
        force=options.force,
    )
    progress.update(task, completed=100, description="Video descargado")
    return media, key, spec


def _analyze(
    media: MediaInfo,
    key: str,
    spec: SourceSpec | None,
    options: RunOptions,
    config: Config,
    cache: Cache,
    progress: Progress,
    *,
    range_start: float = 0.0,
    range_end: float | None = None,
    query_text: str | None = None,
) -> tuple[dict[str, SignalSeries], TranscriptData | None, dict[str, Any]]:
    """Acquire optional inputs and compute (or load) the analysis signals."""
    chat_enabled = config.b("analysis.signals.chat", True)
    audio_enabled = config.b("analysis.signals.audio", True)
    transcript_enabled = config.b("analysis.signals.transcript", True)
    subtitles_enabled = config.b("subtitles.enabled", False)
    query_enabled = config.b("analysis.signals.query", True)
    query_text = (query_text or options.query or "").strip() or None
    range_start = max(0.0, float(range_start))
    range_end = float(media.duration) if range_end is None else min(float(media.duration), float(range_end))
    partial = range_start > 1e-6 or range_end < float(media.duration) - 1e-6
    range_tag = (
        f"r{int(round(range_start * 1000))}-{int(round(range_end * 1000))}"
        if partial
        else ""
    )

    if not media.has_audio:
        if audio_enabled or transcript_enabled or subtitles_enabled or query_text:
            LOG.warning(
                "El video no tiene audio: se desactivan audio, transcripción y subtítulos."
            )
        audio_enabled = False
        transcript_enabled = False
        subtitles_enabled = False
        if not query_text:
            query_enabled = False

    need_transcript = transcript_enabled or subtitles_enabled or query_enabled or bool(query_text)
    need_audio = audio_enabled or need_transcript

    analysis_hash = config.hash("analysis", "weights", "chat", "audio", "transcript")
    if query_text:
        analysis_hash += "_" + hashlib.sha1(
            query_text.encode("utf-8")
        ).hexdigest()[:8]
        if config.b("llm.enabled", False):
            analysis_hash += "l"
    if range_tag:
        analysis_hash += "_" + range_tag
    cache_name = f"signals_{analysis_hash}.json"
    signals: dict[str, SignalSeries] | None = None
    transcript: TranscriptData | None = None
    if not options.force:
        signals = _load_signals_cache(cache, key, cache_name, media.duration)
        if signals:
            LOG.info("Análisis en caché: se reutilizan %d señales.", len(signals))
            if need_transcript:
                transcript = load_cached_transcript(cache, key, config, range_tag)
                if transcript:
                    LOG.info(
                        "Transcripción en caché lista para %s.",
                        "búsqueda" if query_text else ("subtítulos" if subtitles_enabled else "contexto"),
                    )
                else:
                    LOG.info(
                        "No hay transcripción en caché para esta ejecución; "
                        "se recalcula el análisis."
                    )
                    signals = None

    if signals is None:
        wav: Path | None = None
        if need_audio:
            try:
                wav = extract_audio(
                    media,
                    cache,
                    key,
                    config,
                    start=range_start if partial else None,
                    end=range_end if partial else None,
                )
            except ClipHunterError as exc:
                LOG.warning("No se pudo extraer audio: %s", exc)
                audio_enabled = False
                need_transcript = False
                transcript_enabled = False
                subtitles_enabled = False

        executor = ThreadPoolExecutor(max_workers=1)
        chat_future = None
        if chat_enabled:
            if options.chat_file or (spec is not None and options.url):
                chat_spec = spec or SourceSpec(options.url or "", "local", "vod")
                chat_future = executor.submit(
                    fetch_chat,
                    options.url or "",
                    chat_spec,
                    cache,
                    key,
                    config,
                    chat_file=options.chat_file,
                    force=options.force,
                    allow_download=not options.live,
                )
            elif not options.url:
                LOG.info("Archivo local sin --chat-file: se omite la señal de chat.")

        if need_transcript and wav is not None:
            task = progress.add_task("Transcribiendo con Whisper", total=100)
            try:
                transcript = transcribe(
                    wav,
                    cache,
                    key,
                    config,
                    range_end - range_start if partial else media.duration,
                    on_progress=_TaskProgress(progress, task),
                    force=options.force,
                    start=range_start if partial else 0.0,
                    end=range_end if partial else None,
                    tag=range_tag,
                )
            except TranscriptionError as exc:
                LOG.warning("Transcripción no disponible: %s", exc)
                transcript = None
                transcript_enabled = False
                subtitles_enabled = False
            except Exception as exc:  # noqa: BLE001 - never lose the whole run
                LOG.warning("La transcripción falló inesperadamente: %s", exc)
                transcript = None
                transcript_enabled = False
                subtitles_enabled = False
            progress.update(task, completed=100, description="Transcripción terminada")

        messages = None
        if chat_future is not None:
            try:
                messages = chat_future.result()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Chat no disponible: %s", exc)
                messages = None
        executor.shutdown(wait=True)

        signals = {}
        if chat_enabled:
            signals["chat"] = analyze_chat(
                messages, media.duration, config, start=range_start, end=range_end
            )
        if audio_enabled and wav is not None:
            try:
                signals["audio"] = analyze_audio(
                    wav, media.duration, config, start=range_start, end=range_end
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("La señal de audio falló: %s", exc)
        if transcript_enabled and transcript is not None:
            signals["transcript"] = analyze_transcript(
                transcript, media.duration, config, start=range_start, end=range_end
            )

        if query_text and query_enabled and transcript is not None:
            llm_client = LLMClient(config) if config.b("llm.enabled", False) else None
            signals["query"] = analyze_query(
                transcript,
                media.duration,
                config,
                query_text,
                start=range_start,
                end=range_end,
                llm_client=llm_client,
            )

        _save_signals_cache(cache, key, cache_name, signals, media.duration)

    if subtitles_enabled and transcript is None:
        LOG.warning("Subtítulos solicitados pero no hay transcripción disponible.")
        subtitles_enabled = False

    meta = {
        "chat_enabled": chat_enabled,
        "audio_enabled": audio_enabled,
        "transcript_enabled": transcript_enabled,
        "subtitles_enabled": subtitles_enabled,
        "query_enabled": query_enabled,
    }
    return signals, transcript, meta


def _print_summary(result: AnalysisResult, media: MediaInfo, output_dir: Path | None) -> None:
    table = Table(title=f"Mejores momentos — {media.title or media.platform}", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Inicio", style="green", no_wrap=True)
    table.add_column("Fin", style="green", no_wrap=True)
    table.add_column("Dur", justify="right")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Título", overflow="fold")
    table.add_column("Señales", overflow="fold")
    for clip in result.clips:
        signals = ", ".join(
            f"{name} {value:.2f}" for name, value in sorted(clip.breakdown.items())
        )
        table.add_row(
            str(clip.index),
            format_hms(clip.start),
            format_hms(clip.end),
            f"{clip.duration:.0f}s",
            f"{clip.score:.2f}",
            clip.title,
            signals,
        )
    console.print(table)
    if output_dir is not None:
        console.print(f"[dim]Salida:[/] {output_dir}")


def run(options: RunOptions, config: Config) -> PipelineResult:
    """Run the full pipeline and return the exported clips."""
    ensure_media_tools()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(config.cache_dir, enabled=not options.no_cache)
    LOG.info("ClipHunter %s", __version__)
    LOG.info("Caché: %s", cache.root if cache.enabled else "desactivada (temporal)")

    if config.b("llm.enabled", False):
        LLMClient(config).validate()

    with _progress_ui() as progress:
        media, key, spec = _resolve_source(options, config, cache, progress)
        LOG.info(
            "Fuente lista: %s | %s | %s",
            media.title or media.platform,
            format_hms(media.duration),
            media.resolution,
        )

        range_start, range_end = _resolve_analysis_range(options, media.duration)
        signals, transcript, meta = _analyze(
            media,
            key,
            spec,
            options,
            config,
            cache,
            progress,
            range_start=range_start,
            range_end=range_end,
        )
        LOG.info("Señales disponibles: %s", ", ".join(signals) or "ninguna")

        result = score_and_select(
            signals, transcript, media.duration, config, start=range_start, end=range_end
        )

        if config.b("llm.enabled", False) and result.clips:
            try:
                LLMClient(config).rank(result.clips, media)
            except LLMError as exc:
                LOG.warning("LLM no aplicado: %s", exc)

        # Keep the order stable and renumber after a possible LLM reorder.
        for index, clip in enumerate(result.clips, start=1):
            clip.index = index

        if options.dry_run:
            payload = candidate_snapshot(result)
            payload["settings"] = {
                "window": config.f("analysis.window", 10.0),
                "hop": config.f("analysis.hop", 2.0),
                "min_duration": config.f("selection.min_duration", 15.0),
                "max_duration": config.f("selection.max_duration", 60.0),
                "vertical": config.b("export.vertical", False),
                "subtitles": config.b("subtitles.enabled", False),
            }
            payload["times"] = [round(float(t), 2) for t in result.times]
            payload["scores"] = [round(float(s), 4) for s in result.scores]
            path = output_dir / "analysis.json"
            atomic_write_json(path, payload)
            _print_summary(result, media, output_dir)
            console.print(f"[yellow]Dry-run:[/] no se exportaron clips. Análisis: {path}")
            return PipelineResult(media, result.clips, output_dir, None, key)

        if not result.clips:
            LOG.warning("No se encontraron momentos suficientemente buenos.")
            manifest = _write_manifest(output_dir, media, result.clips, config, meta, [])
            _print_summary(result, media, output_dir)
            return PipelineResult(media, result.clips, output_dir, manifest, key)

        vertical = config.b("export.vertical", False)
        subtitles = bool(meta.get("subtitles_enabled")) and transcript is not None
        if config.b("subtitles.enabled", False) and transcript is None:
            LOG.warning("Se exportará sin subtítulos: no hay transcripción.")

        width = 1080 if vertical else max(2, media.width)
        height = 1920 if vertical else max(2, media.height)
        subtitle_files: dict[int, Path | None] = {}
        if subtitles:
            for clip in result.clips:
                target = clip_filename(clip, output_dir)
                ass_path = target.with_suffix(".ass")
                subtitle_files[clip.index] = build_ass(
                    transcript,
                    clip.start,
                    clip.end,
                    ass_path,
                    config=config,
                    vertical=vertical,
                    width=width,
                    height=height,
                )

        workers = max(1, min(config.i("export.workers", 2), len(result.clips)))
        task = progress.add_task("Exportando clips", total=len(result.clips))
        errors: list[dict[str, str]] = []

        def job(clip: CandidateClip) -> CandidateClip:
            export_clip(
                clip,
                media,
                output_dir,
                config,
                vertical=vertical,
                subtitle_path=subtitle_files.get(clip.index),
                # Exports are intentionally always refreshed: output folders are
                # often reused across live sessions with similar clip names.
                overwrite=True,
            )
            return clip

        if workers == 1:
            for clip in result.clips:
                try:
                    job(clip)
                except Exception as exc:  # noqa: BLE001
                    LOG.error("Falló el clip %s: %s", clip.index, exc)
                    errors.append({"index": str(clip.index), "error": str(exc)})
                progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(job, clip): clip for clip in result.clips}
                for future in as_completed(futures):
                    clip = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001
                        LOG.error("Falló el clip %s: %s", clip.index, exc)
                        errors.append({"index": str(clip.index), "error": str(exc)})
                    progress.advance(task)

        manifest_path = _write_manifest(
            output_dir, media, result.clips, config, meta, errors
        )
        _print_summary(result, media, output_dir)
        if errors and len(errors) == len(result.clips):
            raise ClipHunterError(
                "Ningún clip se pudo exportar. Revisa el log; el manifiesto "
                f"con los errores está en {manifest_path}."
            )
        console.print(
            f"[bold green]Listo:[/] {len(result.clips) - len(errors)} clips exportados. "
            f"Manifiesto: {manifest_path}"
        )
        return PipelineResult(media, result.clips, output_dir, manifest_path, key)


def _write_manifest(
    output_dir: Path,
    media: MediaInfo,
    clips: list[CandidateClip],
    config: Config,
    meta: dict[str, Any],
    errors: list[dict[str, str]],
) -> Path:
    payload: dict[str, Any] = {
        "tool": "ClipHunter",
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": media.to_dict(),
        "settings": {
            "clips_requested": config.i("selection.clips", 10),
            "window": config.f("analysis.window", 10.0),
            "hop": config.f("analysis.hop", 2.0),
            "min_duration": config.f("selection.min_duration", 15.0),
            "max_duration": config.f("selection.max_duration", 60.0),
            "pre_roll": config.f("selection.pre_roll", 8.0),
            "post_roll": config.f("selection.post_roll", 4.0),
            "vertical": config.b("export.vertical", False),
            "vertical_mode": config.get("export.vertical_mode", "blur"),
            "subtitles": bool(meta.get("subtitles_enabled")),
            "llm": config.b("llm.enabled", False),
            "signals": {
                "chat": bool(meta.get("chat_enabled")),
                "audio": bool(meta.get("audio_enabled")),
                "transcript": bool(meta.get("transcript_enabled")),
            },
            "weights": config.section("weights"),
        },
        "clips": [clip.to_dict() for clip in clips],
        "errors": errors,
    }
    path = output_dir / "clips.json"
    atomic_write_json(path, payload)
    return path
