"""Source resolution, VOD download, chat acquisition and live recording."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

from .cache import Cache
from .config import Config
from .exceptions import (
    ChatUnavailable,
    ClipHunterError,
    DependencyError,
    LiveOfflineError,
    SourceError,
    UnsupportedSourceError,
)
from .models import ChatMessage, MediaInfo
from .utils import (
    LOG,
    ffmpeg_path,
    format_hms,
    popen_kwargs,
    probe_media,
    run_capture,
    terminate_process_tree,
)

ProgressFn = Callable[[float, str], None]
StatusFn = Callable[[str], None]

SUPPORTED_HELP = (
    "Fuentes soportadas: Twitch (twitch.tv/videos/ID o twitch.tv/canal en directo), "
    "YouTube (watch, shorts, /live o canal en directo) y Kick "
    "(kick.com/video/ID o kick.com/canal en directo)."
)


@dataclass
class SourceSpec:
    """A classified source URL, before metadata resolution."""

    url: str
    platform: str
    kind: str  # "vod" | "live"
    video_id: str = ""
    channel: str = ""


@dataclass
class RemoteSource:
    """Metadata obtained from yt-dlp for a remote source."""

    spec: SourceSpec
    title: str = ""
    duration: float | None = None
    is_live: bool = False
    webpage_url: str = ""
    extractor: str = ""

    @property
    def url(self) -> str:
        return self.webpage_url or self.spec.url

    @property
    def platform(self) -> str:
        return self.spec.platform

    @property
    def video_id(self) -> str:
        return self.spec.video_id or self.spec.channel


# --------------------------------------------------------------------------- URLs
_TWITCH_VOD = re.compile(r"^/videos/(\d+)")
_TWITCH_CHANNEL = re.compile(r"^/([A-Za-z0-9_]{2,25})/?$")
_KICK_VIDEO = re.compile(r"^/video/([0-9a-fA-F-]{8,})")
_KICK_CHANNEL_VIDEO = re.compile(r"^/([A-Za-z0-9_-]+)/videos/([0-9a-fA-F-]{8,})")
_KICK_CHANNEL = re.compile(r"^/([A-Za-z0-9_-]+)/?$")
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def classify_url(url: str) -> SourceSpec:
    """Classify a supported URL without network access."""
    raw = (url or "").strip()
    if not raw:
        raise SourceError("No se indicó ninguna URL.")
    parsed = urlparse(raw if "://" in raw else "https://" + raw)
    if parsed.scheme not in {"http", "https"}:
        raise UnsupportedSourceError(f"Esquema de URL no soportado: {parsed.scheme}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    path = parsed.path or "/"

    if host in {"twitch.tv", "twitch.com"}:
        match = _TWITCH_VOD.match(path)
        if match:
            return SourceSpec(raw, "twitch", "vod", video_id=match.group(1))
        match = _TWITCH_CHANNEL.match(path)
        if match:
            return SourceSpec(raw, "twitch", "live", channel=match.group(1))
        raise UnsupportedSourceError(
            "URL de Twitch no reconocida. Usa un VOD (/videos/ID) o un canal en directo.\n"
            + SUPPORTED_HELP
        )

    if host in {"youtube.com", "youtu.be", "music.youtube.com"}:
        if host == "youtu.be":
            video_id = path.strip("/").split("/")[0]
            if _YT_ID.match(video_id):
                return SourceSpec(raw, "youtube", "vod", video_id=video_id)
        if path.startswith("/watch"):
            video_id = parse_qs(parsed.query).get("v", [""])[0]
            if _YT_ID.match(video_id):
                return SourceSpec(raw, "youtube", "vod", video_id=video_id)
        for prefix in ("/shorts/", "/embed/", "/v/"):
            if path.startswith(prefix):
                video_id = path[len(prefix) :].split("/")[0]
                if _YT_ID.match(video_id):
                    return SourceSpec(raw, "youtube", "vod", video_id=video_id)
        if path.startswith("/live/"):
            video_id = path.split("/")[2]
            return SourceSpec(raw, "youtube", "live", video_id=video_id)
        if path == "/live" or path.endswith("/live"):
            channel = path.strip("/").split("/")[0] if path != "/live" else ""
            return SourceSpec(raw, "youtube", "live", channel=channel.lstrip("@"))
        if path.startswith("/@"):
            channel = path.split("/")[1].lstrip("@")
            return SourceSpec(raw, "youtube", "live", channel=channel)
        raise UnsupportedSourceError(
            "URL de YouTube no reconocida. Usa un enlace watch/shorts o un directo.\n"
            + SUPPORTED_HELP
        )

    if host in {"kick.com"}:
        match = _KICK_CHANNEL_VIDEO.match(path)
        if match:
            return SourceSpec(raw, "kick", "vod", video_id=match.group(2), channel=match.group(1))
        match = _KICK_VIDEO.match(path)
        if match:
            return SourceSpec(raw, "kick", "vod", video_id=match.group(1))
        match = _KICK_CHANNEL.match(path)
        if match and match.group(1) not in {"video", "videos", "categories", "browse"}:
            return SourceSpec(raw, "kick", "live", channel=match.group(1))
        raise UnsupportedSourceError(
            "URL de Kick no reconocida. Usa kick.com/video/ID o un canal en directo.\n"
            + SUPPORTED_HELP
        )

    if re.search(r"\.(mp4|mkv|webm|mov|ts|m3u8)(\?.*)?$", raw, re.IGNORECASE):
        kind = "live" if ".m3u8" in raw.lower() else "vod"
        return SourceSpec(raw, "direct", kind, video_id="")

    raise UnsupportedSourceError(
        f"Plataforma no soportada: {parsed.netloc}\n" + SUPPORTED_HELP
    )


def _cookie_opts(browser: str | None, cookie_file: str | None) -> dict[str, Any]:
    if cookie_file:
        return {"cookiefile": str(Path(cookie_file).expanduser())}
    if browser:
        return {"cookiesfrombrowser": (browser,)}
    return {}


def _clean_ytdlp_error(exc: Exception) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    text = re.sub(r"^ERROR:\s*", "", text.strip())
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else "Error desconocido de yt-dlp"


def _offline_message(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "not currently live",
            "channel is offline",
            "is offline",
            "no playable streams",
            "no video formats",
            "live event will begin",
        )
    )


def fetch_remote_info(
    spec: SourceSpec,
    *,
    cookies_from_browser: str | None = None,
    cookie_file: str | None = None,
) -> RemoteSource:
    """Resolve title/duration/live status using yt-dlp (no download)."""
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise DependencyError(
            "Falta yt-dlp. Instala las dependencias con: pip install -r requirements.txt"
        ) from exc

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "extract_flat": False,
        **_cookie_opts(cookies_from_browser, cookie_file),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(spec.url, download=False)
    except DownloadError as exc:
        message = _clean_ytdlp_error(exc)
        if spec.kind == "live" or _offline_message(message):
            raise LiveOfflineError(
                "El canal no está en directo o el directo no está disponible.\n"
                f"Detalle: {message}"
            ) from exc
        raise SourceError(
            "No se pudo leer el video. Comprueba que el enlace es público.\n"
            f"Detalle: {message}\n"
            "Si es un VOD privado o requiere suscripción, prueba --cookies-from-browser chrome."
        ) from exc

    if not info:
        raise SourceError("yt-dlp no devolvió información para esta URL.")
    if info.get("entries"):
        info = next((e for e in info["entries"] if e), None)
        if not info:
            raise SourceError("La URL no contiene ningún video.")

    duration = info.get("duration")
    return RemoteSource(
        spec=spec,
        title=str(info.get("title") or ""),
        duration=float(duration) if duration else None,
        is_live=bool(info.get("is_live")),
        webpage_url=str(info.get("webpage_url") or spec.url),
        extractor=str(info.get("extractor_key") or ""),
    )


# ----------------------------------------------------------------------- download
def find_media_file(folder: Path) -> Path | None:
    """Pick the downloaded media file (prefer mp4) from a cache folder."""
    candidates = [
        p
        for p in folder.glob("source.*")
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".ts", ".mov"}
        # Skip per-format fragments left by an interrupted yt-dlp merge.
        and not re.match(r"source\.f\d+\.", p.name)
    ]
    if not candidates:
        return None
    priority = {".mp4": 0, ".mkv": 1, ".webm": 2, ".ts": 3, ".mov": 4}
    candidates.sort(key=lambda p: (priority.get(p.suffix.lower(), 9), -p.stat().st_size))
    return candidates[0]


def download_vod(
    remote: RemoteSource,
    cache: Cache,
    key: str,
    *,
    cookies_from_browser: str | None = None,
    cookie_file: str | None = None,
    on_progress: ProgressFn | None = None,
    force: bool = False,
) -> MediaInfo:
    """Download a VOD into the cache and return its metadata."""
    folder = cache.source_dir(key)
    existing = find_media_file(folder)
    if existing and not force:
        LOG.info("Video en caché: %s", existing)
    else:
        if existing and force:
            existing.unlink(missing_ok=True)
        _download_with_ytdlp(
            remote,
            folder,
            cookies_from_browser=cookies_from_browser,
            cookie_file=cookie_file,
            on_progress=on_progress,
        )
        existing = find_media_file(folder)
        if not existing:
            raise SourceError("La descarga terminó pero no se encontró el archivo de video.")

    info = probe_media(existing)
    info.title = remote.title
    info.source_url = remote.url
    info.video_id = remote.video_id
    info.platform = remote.platform
    LOG.info(
        "Video listo: %s (%s, %s, %.1f MB)",
        existing.name,
        format_hms(info.duration),
        info.resolution,
        existing.stat().st_size / (1024 * 1024),
    )
    return info


def _download_with_ytdlp(
    remote: RemoteSource,
    folder: Path,
    *,
    cookies_from_browser: str | None,
    cookie_file: str | None,
    on_progress: ProgressFn | None,
) -> None:
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise DependencyError(
            "Falta yt-dlp. Instala las dependencias con: pip install -r requirements.txt"
        ) from exc

    def hook(data: dict[str, Any]) -> None:
        if on_progress is None:
            return
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            done = data.get("downloaded_bytes") or 0
            fraction = float(done) / float(total) if total else 0.0
            speed = data.get("_speed_str") or ""
            on_progress(min(0.99, fraction), f"Descargando video {speed}".strip())
        elif status == "finished":
            on_progress(0.99, "Procesando video descargado")

    class _QuietLogger:
        def debug(self, msg: str) -> None:
            LOG.debug("yt-dlp: %s", msg)

        def warning(self, msg: str) -> None:
            LOG.debug("yt-dlp warning: %s", msg)

        def error(self, msg: str) -> None:
            LOG.debug("yt-dlp error: %s", msg)

    opts: dict[str, Any] = {
        "outtmpl": str(folder / "source.%(ext)s"),
        "format": (
            "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/"
            "bv*[height<=1080]+ba/b[height<=1080]/b"
        ),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 30,
        "progress_hooks": [hook],
        "logger": _QuietLogger(),
        "overwrites": False,
        "continuedl": True,
        **_cookie_opts(cookies_from_browser, cookie_file),
    }
    LOG.info("Descargando %s (%s)", remote.title or remote.url, remote.platform)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([remote.url])
    except DownloadError as exc:
        message = _clean_ytdlp_error(exc)
        if _offline_message(message):
            raise LiveOfflineError(f"El directo no está disponible: {message}") from exc
        raise SourceError(
            "Falló la descarga con yt-dlp.\n"
            f"Detalle: {message}\n"
            "Prueba a actualizar yt-dlp (pip install -U yt-dlp) o usar --cookies-from-browser."
        ) from exc


# --------------------------------------------------------------------------- chat
def _normalize_comment(raw: Any, *, live_elapsed: float | None = None) -> ChatMessage | None:
    """Convert a chat-downloader comment / TwitchDownloader comment to ChatMessage."""
    if not isinstance(raw, dict):
        return None
    message = raw.get("message", raw.get("text", raw.get("body", "")))
    if isinstance(message, dict):
        body = message.get("body") or message.get("message") or ""
        raw_emotes = message.get("emotes") or raw.get("emotes") or []
    else:
        body = str(message or "")
        raw_emotes = raw.get("emotes") or []
    body = body.strip()
    if not body:
        return None

    if live_elapsed is not None:
        timestamp = live_elapsed
    else:
        timestamp = raw.get("time_in_seconds")
        if timestamp is None:
            timestamp = raw.get("content_offset_seconds")
        if timestamp is None:
            timestamp = raw.get("video_offset_seconds")
        if timestamp is None:
            timestamp = raw.get("t", raw.get("time"))
    if timestamp is None:
        return None
    try:
        t = float(timestamp)
    except (TypeError, ValueError):
        return None

    emote_names: list[str] = []
    if isinstance(raw_emotes, dict):
        raw_emotes = list(raw_emotes.keys())
    if isinstance(raw_emotes, list):
        for emote in raw_emotes:
            if isinstance(emote, str):
                emote_names.append(emote)
            elif isinstance(emote, dict):
                name = emote.get("name") or emote.get("text") or emote.get("id")
                if name:
                    emote_names.append(str(name))
    author = raw.get("author")
    author_name = ""
    if isinstance(author, dict):
        author_name = str(author.get("name") or author.get("display_name") or "")
    elif isinstance(author, str):
        author_name = author

    return ChatMessage(t=max(0.0, t), text=body, emotes=emote_names, author=author_name)


def _load_chat_file(path: Path) -> list[ChatMessage]:
    """Load chat from JSON, TwitchDownloader JSON or JSONL."""
    if not path.is_file():
        raise ChatUnavailable(f"No existe el archivo de chat: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    raw_items: list[Any] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            raw_items = data.get("comments") or data.get("messages") or []
        elif isinstance(data, list):
            raw_items = data
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    messages: list[ChatMessage] = []
    for item in raw_items:
        msg = _normalize_comment(item)
        if msg:
            messages.append(msg)
    messages.sort(key=lambda m: m.t)
    return messages


def _chat_cache_payload(messages: Iterable[ChatMessage]) -> list[dict[str, Any]]:
    return [
        {"t": round(m.t, 3), "text": m.text, "emotes": m.emotes, "author": m.author}
        for m in messages
    ]


def _messages_from_payload(payload: Any) -> list[ChatMessage]:
    if not isinstance(payload, list):
        return []
    result: list[ChatMessage] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            t = float(item.get("t", item.get("time", 0.0)))
        except (TypeError, ValueError):
            continue
        result.append(
            ChatMessage(
                t=t,
                text=str(item.get("text", item.get("message", ""))),
                emotes=[str(e) for e in (item.get("emotes") or [])],
                author=str(item.get("author", "")),
            )
        )
    return result


def _run_with_timeout(fn: Callable[[], Any], timeout: float) -> Any:
    """Run blocking network code in a daemon thread with a hard timeout."""
    result: list[Any] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(fn())
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise ChatUnavailable(f"La descarga del chat superó {int(timeout)} s.")
    if error:
        raise ChatUnavailable(str(error[0]))
    return result[0] if result else []


def _twitchdownloader_executable() -> str | None:
    env = os.environ.get("CLIPHUNTER_TWITCHDOWNLOADER")
    if env and Path(env).is_file():
        return env
    return shutil.which("TwitchDownloaderCLI") or shutil.which("TwitchDownloaderCLI.exe")


def _fetch_chat_twitchdownloader(spec: SourceSpec, folder: Path) -> list[ChatMessage]:
    exe = _twitchdownloader_executable()
    if not exe or not spec.video_id:
        raise ChatUnavailable("TwitchDownloaderCLI no está instalado.")
    output = folder / "chat_twitchdownloader.json"
    LOG.info("Descargando chat con TwitchDownloaderCLI…")
    run_capture(
        [exe, "chatdownload", "--id", spec.video_id, "-o", str(output)],
        timeout=900,
    )
    return _load_chat_file(output)


def fetch_chat(
    url: str,
    spec: SourceSpec,
    cache: Cache,
    key: str,
    config: Config,
    *,
    chat_file: str | Path | None = None,
    on_progress: Callable[[int], None] | None = None,
    force: bool = False,
    allow_download: bool = True,
) -> list[ChatMessage] | None:
    """Download chat for a VOD/live URL. Returns ``None`` when unavailable.

    ``allow_download=False`` only reads the local cache (used after recording a
    live stream, where chat was already captured in parallel).
    """
    if chat_file:
        messages = _load_chat_file(Path(chat_file).expanduser())
        cache.save_json(key, "chat.json", _chat_cache_payload(messages))
        LOG.info("Chat cargado desde archivo local: %d mensajes", len(messages))
        return messages

    if not force:
        cached = cache.load_json(key, "chat.json")
        if cached is not None:
            messages = _messages_from_payload(cached)
            if messages:
                LOG.info("Chat en caché: %d mensajes", len(messages))
                return messages
            LOG.warning("La caché de chat está vacía; se intentará descargar de nuevo.")

    if not allow_download:
        return None

    if not force:
        unavailable = cache.load_json(key, "chat_unavailable.json")
        if unavailable:
            LOG.info("Chat no disponible (en caché): %s", unavailable.get("reason", ""))
            return None

    if spec.platform == "kick":
        reason = "Kick no ofrece una API pública de chat para VODs."
        cache.save_json(key, "chat_unavailable.json", {"reason": reason})
        LOG.warning("%s Se continúa sin la señal de chat.", reason)
        return None

    if not force:
        cached = cache.load_json(key, "chat_downloader.json")
        if cached is not None:
            messages = _messages_from_payload(cached)
            if messages:
                cache.save_json(key, "chat.json", _chat_cache_payload(messages))
                return messages

    timeout = config.f("chat.timeout", 600.0)
    max_messages = config.i("chat.max_messages", 500_000)

    def collect() -> list[ChatMessage]:
        from chat_downloader import ChatDownloader

        downloader = ChatDownloader()
        result: list[ChatMessage] = []
        generator = downloader.get_chat(
            url,
            message_groups=["messages", "emotes"],
            timeout=int(timeout),
            max_messages=max_messages,
        )
        for index, comment in enumerate(generator):
            message = _normalize_comment(comment)
            if message is not None:
                result.append(message)
            if on_progress and index and index % 500 == 0:
                on_progress(index)
        return result

    LOG.info("Descargando chat de %s…", spec.platform)
    try:
        messages = _run_with_timeout(collect, timeout + 30)
    except ChatUnavailable as exc:
        LOG.warning("No se pudo descargar el chat: %s", exc)
        if spec.platform == "twitch":
            try:
                messages = _fetch_chat_twitchdownloader(spec, cache.source_dir(key))
            except (ChatUnavailable, ClipHunterError) as fallback_exc:
                LOG.warning("TwitchDownloaderCLI tampoco pudo: %s", fallback_exc)
                messages = []
        else:
            messages = []
    except Exception as exc:  # noqa: BLE001 - chat-downloader raises many types
        LOG.warning("No se pudo descargar el chat: %s", exc)
        messages = []

    if not messages:
        reason = "No se encontraron mensajes de chat para este video."
        cache.save_json(key, "chat_unavailable.json", {"reason": reason})
        LOG.warning("%s", reason)
        return None

    messages.sort(key=lambda m: m.t)
    cache.save_json(key, "chat.json", _chat_cache_payload(messages))
    LOG.info("Chat descargado: %d mensajes", len(messages))
    return messages


# --------------------------------------------------------------------------- local
def local_source(path: str | Path) -> MediaInfo:
    """Validate a local media file and return its metadata."""
    media = Path(path).expanduser()
    if not media.is_file():
        raise SourceError(f"No existe el archivo local: {media}")
    return probe_media(media)


# ---------------------------------------------------------------------------- live
class LiveRecorder:
    """Record a live stream to segmented MPEG-TS and remux it to MP4.

    streamlink (or yt-dlp as fallback) writes MPEG-TS to stdout; ffmpeg splits
    it into 30 s segments. The last segment is finalized when the source closes
    stdout. Chat is recorded in parallel whenever chat-downloader supports it.
    """

    SEGMENT_SECONDS = 30

    def __init__(self, cache: Cache, key: str, config: Config) -> None:
        self.cache = cache
        self.key = key
        self.config = config
        self.folder = cache.source_dir(key)
        self._stop = threading.Event()
        self._logs: dict[str, deque[str]] = {
            "streamlink": deque(maxlen=120),
            "ffmpeg": deque(maxlen=120),
        }
        self._chat_thread: threading.Thread | None = None
        self._chat_stop = threading.Event()
        self._chat_start = 0.0
        self._chat_count = 0

    def request_stop(self) -> None:
        """Ask the recording loop to stop (used by Ctrl+C)."""
        self._stop.set()

    # ---------------------------------------------------------------- utilities
    @staticmethod
    def _reader(stream: Any, sink: deque[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                text = line.strip()
                if text:
                    sink.append(text)
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    @staticmethod
    def _streamlink_prefix() -> list[str] | None:
        env = os.environ.get("CLIPHUNTER_STREAMLINK")
        if env and Path(env).is_file():
            return [env]
        found = shutil.which("streamlink")
        if found:
            return [found]
        # Running from a venv does not put Scripts/ on PATH for child processes.
        sibling = Path(sys.executable).with_name(
            "streamlink.exe" if os.name == "nt" else "streamlink"
        )
        if sibling.is_file():
            return [str(sibling)]
        try:
            import streamlink  # noqa: F401

            return [sys.executable, "-m", "streamlink"]
        except ImportError:
            return None

    def _source_command(self, url: str) -> tuple[list[str], str]:
        streamlink = self._streamlink_prefix()
        if streamlink:
            cmd = [
                *streamlink,
                "--twitch-disable-ads",
                "--hls-live-edge",
                "3",
                "--stream-timeout",
                "45",
                "--stdout",
                url,
                "best",
            ]
            return cmd, "streamlink"
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-warnings",
            "--quiet",
            "-f",
            "best",
            "-o",
            "-",
            url,
        ]
        return cmd, "yt-dlp"

    def _start_chat(self, url: str) -> None:
        if not self.config.b("analysis.signals.chat", True):
            return
        if "kick.com" in url.lower():
            LOG.warning("Kick no ofrece chat en directo vía chat-downloader; se omite.")
            return

        chat_file = self.folder / "live_chat.jsonl"

        def worker() -> None:
            try:
                from chat_downloader import ChatDownloader

                downloader = ChatDownloader()
                generator = downloader.get_chat(url, message_groups=["messages", "emotes"])
                with chat_file.open("w", encoding="utf-8") as fh:
                    for comment in generator:
                        if self._chat_stop.is_set():
                            break
                        message = _normalize_comment(
                            comment, live_elapsed=time.monotonic() - self._chat_start
                        )
                        if message is None:
                            continue
                        fh.write(
                            json.dumps(
                                {
                                    "t": round(message.t, 3),
                                    "text": message.text,
                                    "emotes": message.emotes,
                                    "author": message.author,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        fh.flush()
                        self._chat_count += 1
            except Exception as exc:  # noqa: BLE001 - chat is best effort
                LOG.warning("Chat en directo no disponible: %s", exc)

        self._chat_start = time.monotonic()
        self._chat_thread = threading.Thread(target=worker, daemon=True)
        self._chat_thread.start()

    def _stop_chat(self) -> None:
        self._chat_stop.set()
        if self._chat_thread and self._chat_thread.is_alive():
            self._chat_thread.join(timeout=5)
        chat_file = self.folder / "live_chat.jsonl"
        if chat_file.is_file() and chat_file.stat().st_size > 0:
            messages = _load_chat_file(chat_file)
            if messages:
                self.cache.save_json(self.key, "chat.json", _chat_cache_payload(messages))
                LOG.info("Chat en directo: %d mensajes", len(messages))
            else:
                LOG.warning("El chat en directo no produjo mensajes utilizables.")

    # ------------------------------------------------------------------- record
    def record(
        self,
        url: str,
        max_minutes: float,
        *,
        on_status: StatusFn | None = None,
        on_progress: ProgressFn | None = None,
        record_chat: bool = True,
    ) -> MediaInfo:
        """Record until ``max_minutes`` (or Ctrl+C) and return MP4 metadata."""
        for old in self.folder.glob("seg_*.ts"):
            old.unlink(missing_ok=True)
        for name in ("source.mp4", "live_chat.jsonl", "chat.json"):
            (self.folder / name).unlink(missing_ok=True)

        cmd, backend = self._source_command(url)
        LOG.info(
            "Grabando en directo con %s durante %s (segmentos de %ds)…",
            backend,
            f"{max_minutes:.1f} min" if max_minutes > 0 else "hasta Ctrl+C",
            self.SEGMENT_SECONDS,
        )
        if record_chat:
            self._start_chat(url)

        source = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs(),
        )
        assert source.stdout is not None and source.stderr is not None
        threading.Thread(
            target=self._reader, args=(source.stderr, self._logs["streamlink"]), daemon=True
        ).start()

        ffmpeg = subprocess.Popen(
            [
                ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "+genpts",
                "-i",
                "pipe:0",
                "-c",
                "copy",
                "-f",
                "segment",
                "-segment_time",
                str(self.SEGMENT_SECONDS),
                "-reset_timestamps",
                "1",
                "-segment_format",
                "mpegts",
                str(self.folder / "seg_%04d.ts"),
            ],
            stdin=source.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs(),
        )
        assert ffmpeg.stderr is not None
        threading.Thread(
            target=self._reader, args=(ffmpeg.stderr, self._logs["ffmpeg"]), daemon=True
        ).start()
        # The parent must not keep the pipe open, otherwise ffmpeg never sees EOF.
        source.stdout.close()

        started = time.monotonic()
        limit = max_minutes * 60 if max_minutes > 0 else float("inf")
        try:
            while True:
                elapsed = time.monotonic() - started
                if on_progress:
                    fraction = min(1.0, elapsed / limit) if limit != float("inf") else 0.0
                    on_progress(fraction, f"Grabando en directo {format_hms(elapsed)}")
                if self._stop.is_set():
                    LOG.info("Deteniendo grabación (Ctrl+C)…")
                    break
                if elapsed >= limit:
                    LOG.info("Duración máxima alcanzada (%s).", f"{max_minutes:.1f} min")
                    break
                if ffmpeg.poll() is not None:
                    LOG.error("ffmpeg terminó inesperadamente durante la grabación.")
                    break
                if source.poll() is not None:
                    LOG.warning("%s terminó durante la grabación.", backend)
                    break
                time.sleep(0.5)
        finally:
            self._stop_recording(source, ffmpeg)
            if record_chat:
                self._stop_chat()

        segments = sorted(self.folder.glob("seg_*.ts"))
        if not segments:
            detail = "\n".join(list(self._logs["streamlink"])[-6:])
            if _offline_message(detail):
                raise LiveOfflineError(
                    "El canal no está en directo (streamlink no encontró stream).\n" + detail
                )
            raise LiveOfflineError(
                "No se grabó ningún segmento. Comprueba que el directo está activo.\n" + detail
            )

        output = self._concat_segments(segments)
        info = probe_media(output)
        info.title = f"Directo {url}"
        info.source_url = url
        info.platform = "live"
        LOG.info(
            "Grabación lista: %s (%s, %.1f MB)",
            output.name,
            format_hms(info.duration),
            output.stat().st_size / (1024 * 1024),
        )
        return info

    def _stop_recording(self, source: subprocess.Popen[Any], ffmpeg: subprocess.Popen[Any]) -> None:
        # Close the source (and its children) first so ffmpeg receives EOF and
        # finalizes the last segment. Killing only yt-dlp can leave its ffmpeg
        # child holding the pipe open for a long time on Windows.
        LOG.info("Cerrando la grabación y finalizando segmentos…")
        terminate_process_tree(source, timeout=10)
        try:
            ffmpeg.wait(timeout=20)
        except subprocess.TimeoutExpired:
            LOG.warning("ffmpeg no cerró a tiempo; se fuerza la parada.")
            terminate_process_tree(ffmpeg, timeout=5)

    def _concat_segments(self, segments: list[Path]) -> Path:
        list_file = self.folder / "segments.txt"
        list_file.write_text(
            "\n".join(f"file '{seg.resolve().as_posix()}'" for seg in segments) + "\n",
            encoding="utf-8",
        )
        output = self.folder / "source.mp4"
        LOG.info("Uniendo %d segmentos en %s…", len(segments), output.name)
        run_capture(
            [
                ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                "-bsf:a",
                "aac_adtstoasc",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise ClipHunterError("No se pudo unir los segmentos del directo.")
        return output
