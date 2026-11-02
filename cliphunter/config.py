"""Configuration loading, merging, environment overrides and validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .exceptions import ClipHunterError

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

_ENV_OVERRIDES = {
    "paths.cache_dir": "CLIPHUNTER_CACHE_DIR",
    "llm.base_url": "CLIPHUNTER_LLM_BASE_URL",
    "llm.api_key": "CLIPHUNTER_LLM_API_KEY",
    "llm.model": "CLIPHUNTER_LLM_MODEL",
    "transcript.model": "CLIPHUNTER_WHISPER_MODEL",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Thin, typed-ish wrapper around the YAML configuration tree."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    # ------------------------------------------------------------------ access
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ClipHunterError(f"Config path {dotted!r} is not a mapping")
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ClipHunterError(f"Config section {name!r} must be a mapping")
        return value

    def f(self, dotted: str, default: float = 0.0) -> float:
        value = self.get(dotted, default)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ClipHunterError(f"Config value {dotted!r} must be a number") from exc

    def i(self, dotted: str, default: int = 0) -> int:
        return int(self.f(dotted, float(default)))

    def b(self, dotted: str, default: bool = False) -> bool:
        value = self.get(dotted, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    # ------------------------------------------------------------------- misc
    @property
    def output_dir(self) -> Path:
        return Path(str(self.get("paths.output_dir", "clips"))).expanduser()

    @property
    def cache_dir(self) -> Path | None:
        raw = self.get("paths.cache_dir")
        return Path(str(raw)).expanduser() if raw else None

    def apply_env(self) -> None:
        """Apply supported environment variables without ever hardcoding keys."""
        load_dotenv()
        for dotted, env_name in _ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value:
                self.set(dotted, value)

    def validate(self) -> None:
        hop = self.f("analysis.hop", 2.0)
        window = self.f("analysis.window", 10.0)
        if hop <= 0:
            raise ClipHunterError("analysis.hop must be > 0")
        if window < hop:
            raise ClipHunterError("analysis.window must be >= analysis.hop")

        min_dur = self.f("selection.min_duration", 15.0)
        max_dur = self.f("selection.max_duration", 60.0)
        if min_dur <= 0 or max_dur < min_dur:
            raise ClipHunterError("selection durations must satisfy 0 < min_duration <= max_duration")
        if self.i("selection.clips", 10) < 1:
            raise ClipHunterError("selection.clips must be >= 1")

        weights = self.section("weights")
        numeric = {k: float(v) for k, v in weights.items() if isinstance(v, (int, float))}
        if not numeric or sum(numeric.values()) <= 0:
            raise ClipHunterError("At least one signal weight must be > 0")
        if any(v < 0 for v in numeric.values()):
            raise ClipHunterError("Signal weights cannot be negative")

        weight = self.f("llm.weight", 0.3)
        if not 0.0 <= weight <= 1.0:
            raise ClipHunterError("llm.weight must be between 0 and 1")

    def hash(self, *sections: str) -> str:
        """Stable hash of selected sections, used as an analysis cache key."""
        payload: dict[str, Any] = {}
        for name in sections:
            payload[name] = self.data.get(name)
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe config copy with secrets redacted."""
        data = copy.deepcopy(self.data)
        if data.get("llm", {}).get("api_key"):
            data["llm"]["api_key"] = "***"
        return data


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load defaults, merge an optional user YAML file and CLI overrides."""
    try:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise ClipHunterError(f"Cannot read default config: {DEFAULT_CONFIG_PATH}") from exc

    if path:
        user_path = Path(path).expanduser()
        if not user_path.is_file():
            raise ClipHunterError(f"Config file not found: {user_path}")
        try:
            with user_path.open("r", encoding="utf-8") as fh:
                user_data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ClipHunterError(f"Invalid YAML in {user_path}: {exc}") from exc
        if not isinstance(user_data, dict):
            raise ClipHunterError(f"Config file must contain a mapping: {user_path}")
        data = _deep_merge(data, user_data)

    cfg = Config(data)
    # Environment first, then explicit CLI overrides: command-line values win.
    cfg.apply_env()
    for dotted, value in (overrides or {}).items():
        if value is not None:
            cfg.set(dotted, value)

    cfg.validate()
    return cfg
