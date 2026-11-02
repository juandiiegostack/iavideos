"""Persistent cache for downloads, audio, chat and analysis results."""

from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .utils import LOG, atomic_write_json, read_json


def default_cache_root() -> Path:
    """Return a per-user cache directory for the current OS."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return Path(base) / "ClipHunter" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ClipHunter"
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "cliphunter"


def _clean_fragment(text: str, max_len: int = 48) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:max_len] or "source"


def source_key(
    *,
    platform: str = "",
    video_id: str = "",
    url: str = "",
    local_path: str | Path | None = None,
) -> str:
    """Build a stable, human-readable cache key for a source."""
    if local_path is not None:
        path = Path(local_path).expanduser().resolve()
        stat = path.stat()
        raw = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        return f"local_{_clean_fragment(path.stem, 32)}_{digest}"
    if platform and video_id:
        return f"{_clean_fragment(platform.lower(), 16)}_{_clean_fragment(video_id, 48)}"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"url_{digest}"


class Cache:
    """Small directory-backed cache; disabled mode uses a temp dir for one run."""

    def __init__(self, root: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        if not enabled:
            self._tempdir = tempfile.TemporaryDirectory(prefix="cliphunter-")
            base = Path(self._tempdir.name)
            atexit.register(self._cleanup)
        else:
            base = Path(root).expanduser() if root else default_cache_root()
        self.root = base
        self.root.mkdir(parents=True, exist_ok=True)

    def _cleanup(self) -> None:
        if self._tempdir is not None:
            try:
                self._tempdir.cleanup()
            except OSError:
                pass

    def source_dir(self, key: str) -> Path:
        folder = self.root / _clean_fragment(key, 96)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def path(self, key: str, filename: str) -> Path:
        return self.source_dir(key) / filename

    def exists(self, key: str, filename: str) -> bool:
        path = self.path(key, filename)
        return path.is_file() and path.stat().st_size > 0

    def load_json(self, key: str, filename: str) -> Any | None:
        path = self.path(key, filename)
        if not path.is_file():
            return None
        try:
            return read_json(path)
        except (OSError, ValueError) as exc:
            LOG.warning("Caché corrupta, se ignora %s: %s", path, exc)
            return None

    def save_json(self, key: str, filename: str, payload: Any) -> Path:
        path = self.path(key, filename)
        atomic_write_json(path, payload)
        return path

    def remove(self, key: str, filename: str) -> None:
        try:
            self.path(key, filename).unlink(missing_ok=True)
        except OSError:
            pass

    def clear(self, key: str | None = None) -> None:
        target = self.source_dir(key) if key else self.root
        if target == self.root:
            for child in self.root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        else:
            shutil.rmtree(target, ignore_errors=True)
        LOG.info("Caché borrada: %s", target)
