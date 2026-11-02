"""Command line interface for ClipHunter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config
from .exceptions import ClipHunterError
from .pipeline import RunOptions, run
from .utils import LOG, console, setup_logging

EXAMPLES = """\
Ejemplos:
  python cliphunter.py "https://twitch.tv/videos/123456789" --clips 10
  python cliphunter.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --vertical --subtitles
  python cliphunter.py "https://twitch.tv/canal" --live --max-duration 20 --clips 5
  python cliphunter.py --file partida.mp4 --chat-file chat.json --no-audio
  python cliphunter.py "https://kick.com/video/..." --clips 3 --dry-run
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cliphunter.py",
        description=(
            "Detecta los mejores momentos de un VOD o directo y exporta clips "
            "listos para publicar."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument("url", nargs="?", help="URL del VOD o directo (Twitch, YouTube, Kick).")
    parser.add_argument("--file", help="Archivo local de video para probar sin descargar.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Grabar un directo (requiere --max-duration) y analizarlo al terminar.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        metavar="MIN",
        help="Minutos a grabar en modo directo (0 = hasta Ctrl+C). Por defecto: 5.",
    )
    parser.add_argument("--clips", type=int, help="Número de clips a exportar (defecto 10).")
    parser.add_argument("--min-dur", type=float, help="Duración mínima del clip en segundos.")
    parser.add_argument("--max-dur", type=float, help="Duración máxima del clip en segundos.")
    parser.add_argument("--pre-roll", type=float, help="Segundos antes del pico (defecto 8).")
    parser.add_argument("--post-roll", type=float, help="Segundos después del pico (defecto 4).")
    parser.add_argument("--window", type=float, help="Ventana de análisis en segundos (defecto 10).")
    parser.add_argument("--hop", type=float, help="Paso entre ventanas en segundos (defecto 2).")
    parser.add_argument("--out", metavar="DIR", help="Carpeta de salida (defecto clips/).")
    parser.add_argument("--config", metavar="YAML", help="Archivo de configuración alternativo.")

    signals = parser.add_argument_group("señales")
    chat = signals.add_mutually_exclusive_group()
    chat.add_argument("--chat", dest="chat", action="store_true", help="Activar la señal de chat.")
    chat.add_argument("--no-chat", dest="chat", action="store_false", help="Desactivar el chat.")
    audio = signals.add_mutually_exclusive_group()
    audio.add_argument("--audio", dest="audio", action="store_true", help="Activar la señal de audio.")
    audio.add_argument("--no-audio", dest="audio", action="store_false", help="Desactivar el audio.")
    transcript = signals.add_mutually_exclusive_group()
    transcript.add_argument(
        "--transcript", dest="transcript", action="store_true", help="Activar la transcripción."
    )
    transcript.add_argument(
        "--no-transcript", dest="transcript", action="store_false", help="Desactivar Whisper."
    )
    parser.set_defaults(chat=None, audio=None, transcript=None)

    parser.add_argument("--model", help="Modelo Whisper: tiny, base, small, medium, large-v3…")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], help="Dispositivo para Whisper."
    )
    parser.add_argument("--language", help="Idioma para Whisper (es, en…). Defecto: autodetectar.")
    parser.add_argument(
        "--chat-file", metavar="JSON",
        help="Chat local (JSON/JSONL) para archivos locales o para forzar otro chat.",
    )

    output = parser.add_argument_group("formato")
    vertical = output.add_mutually_exclusive_group()
    vertical.add_argument(
        "--vertical", dest="vertical", action="store_true", help="Exportar en 9:16 (1080x1920)."
    )
    vertical.add_argument(
        "--no-vertical", dest="vertical", action="store_false", help="Exportar horizontal."
    )
    output.add_argument(
        "--vertical-mode", choices=["blur", "crop"],
        help="Modo vertical: fondo desenfocado o recorte centrado.",
    )
    subtitles = output.add_mutually_exclusive_group()
    subtitles.add_argument(
        "--subtitles", dest="subtitles", action="store_true", help="Quemar subtítulos estilo TikTok."
    )
    subtitles.add_argument(
        "--no-subtitles", dest="subtitles", action="store_false", help="No quemar subtítulos."
    )
    output.add_argument(
        "--subtitle-mode", choices=["word", "chunk"], help="Subtítulos palabra a palabra o por frases."
    )
    parser.set_defaults(vertical=None, subtitles=None)

    llm = output.add_mutually_exclusive_group()
    llm.add_argument("--llm", dest="llm", action="store_true", help="Reordenar con un LLM opcional.")
    llm.add_argument("--no-llm", dest="llm", action="store_false", help="Desactivar el LLM.")
    parser.set_defaults(llm=None)

    parser.add_argument("--workers", type=int, help="Exportaciones ffmpeg en paralelo (defecto 2).")
    parser.add_argument("--dry-run", action="store_true", help="Analizar y mostrar momentos sin exportar.")

    cache = parser.add_argument_group("caché y red")
    cache.add_argument("--cache-dir", metavar="DIR", help="Carpeta de caché alternativa.")
    cache.add_argument("--no-cache", action="store_true", help="No reutilizar ni conservar caché.")
    cache.add_argument(
        "--clear-cache", action="store_true", help="Borrar la caché de esta fuente antes de empezar."
    )
    cache.add_argument("--force", action="store_true", help="Recalcular y sobrescribir todo.")
    cache.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Usar cookies de un navegador para VODs privados (chrome, firefox, edge…).",
    )
    cache.add_argument("--cookies", metavar="FILE", help="Archivo cookies.txt para yt-dlp.")

    parser.add_argument("--verbose", action="store_true", help="Logs detallados.")
    parser.add_argument("--version", action="version", version=f"ClipHunter {__version__}")
    return parser


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.out:
        overrides["paths.output_dir"] = args.out
    if args.cache_dir:
        overrides["paths.cache_dir"] = args.cache_dir
    if args.clips is not None:
        overrides["selection.clips"] = args.clips
    if args.min_dur is not None:
        overrides["selection.min_duration"] = args.min_dur
    if args.max_dur is not None:
        overrides["selection.max_duration"] = args.max_dur
    if args.pre_roll is not None:
        overrides["selection.pre_roll"] = args.pre_roll
    if args.post_roll is not None:
        overrides["selection.post_roll"] = args.post_roll
    if args.window is not None:
        overrides["analysis.window"] = args.window
    if args.hop is not None:
        overrides["analysis.hop"] = args.hop
    if args.chat is not None:
        overrides["analysis.signals.chat"] = args.chat
    if args.audio is not None:
        overrides["analysis.signals.audio"] = args.audio
    if args.transcript is not None:
        overrides["analysis.signals.transcript"] = args.transcript
    if args.model:
        overrides["transcript.model"] = args.model
    if args.device:
        overrides["transcript.device"] = args.device
    if args.language:
        overrides["transcript.language"] = args.language
    if args.vertical is not None:
        overrides["export.vertical"] = args.vertical
    if args.vertical_mode:
        overrides["export.vertical_mode"] = args.vertical_mode
    if args.subtitles is not None:
        overrides["subtitles.enabled"] = args.subtitles
    if args.subtitle_mode:
        overrides["subtitles.mode"] = args.subtitle_mode
    if args.llm is not None:
        overrides["llm.enabled"] = args.llm
    if args.workers is not None:
        overrides["export.workers"] = args.workers
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if bool(args.url) == bool(args.file):
        parser.error("Indica exactamente una fuente: una URL o --file ruta.mp4.")
    if args.live and args.file:
        parser.error("--live solo funciona con una URL de directo, no con --file.")
    if args.max_duration < 0:
        parser.error("--max-duration no puede ser negativo (0 = hasta Ctrl+C).")
    if args.clips is not None and args.clips < 1:
        parser.error("--clips debe ser >= 1.")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers debe ser >= 1.")

    try:
        config = load_config(args.config, _config_overrides(args))
        options = RunOptions(
            url=args.url,
            file=args.file,
            live=args.live,
            max_duration=args.max_duration,
            chat_file=args.chat_file,
            cookies_from_browser=args.cookies_from_browser,
            cookie_file=args.cookies,
            no_cache=args.no_cache,
            clear_cache=args.clear_cache,
            force=args.force,
            dry_run=args.dry_run,
        )
        result = run(options, config)
    except KeyboardInterrupt:
        console.print("[yellow]Cancelado por el usuario.[/]")
        return 130
    except ClipHunterError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        if args.verbose:
            LOG.exception("Detalle del error")
        return 1
    except Exception as exc:  # noqa: BLE001 - friendly top-level handler
        console.print(f"[bold red]Error inesperado:[/] {exc}")
        if args.verbose:
            raise
        console.print("[dim]Repite con --verbose para ver la traza completa.[/]")
        return 1

    if result.manifest_path:
        console.print(f"[dim]clips.json:[/] {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
