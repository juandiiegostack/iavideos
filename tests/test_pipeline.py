from __future__ import annotations

from pathlib import Path

import pytest

from cliphunter.cache import Cache
from cliphunter.config import load_config
from cliphunter.downloader import RemoteSource, SourceSpec
from cliphunter.exceptions import SourceError
from cliphunter.pipeline import RunOptions, _resolve_source


class DummyProgress:
    def add_task(self, *args: object, **kwargs: object) -> int:
        return 0

    def update(self, *args: object, **kwargs: object) -> None:
        return None


def test_live_mode_rejects_a_non_live_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cliphunter.pipeline as pipeline_module

    spec = SourceSpec("https://www.youtube.com/watch?v=abc12345678", "youtube", "vod", video_id="abc12345678")
    monkeypatch.setattr(pipeline_module, "classify_url", lambda url: spec)
    monkeypatch.setattr(
        pipeline_module,
        "fetch_remote_info",
        lambda *args, **kwargs: RemoteSource(spec=spec, title="Video", duration=120, is_live=False),
    )

    options = RunOptions(url=spec.url, live=True, max_duration=1)
    config = load_config()
    cache = Cache(tmp_path / "cache", enabled=False)
    with pytest.raises(SourceError) as info:
        _resolve_source(options, config, cache, DummyProgress())
    assert "no es un directo" in str(info.value)


def test_live_mode_offline_channel_reports_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cliphunter.pipeline as pipeline_module
    from cliphunter.exceptions import LiveOfflineError

    spec = SourceSpec("https://twitch.tv/canal", "twitch", "live", channel="canal")
    monkeypatch.setattr(pipeline_module, "classify_url", lambda url: spec)

    def offline(*args: object, **kwargs: object) -> RemoteSource:
        raise LiveOfflineError("El canal no está en directo")

    monkeypatch.setattr(pipeline_module, "fetch_remote_info", offline)

    options = RunOptions(url=spec.url, live=True, max_duration=1)
    cache = Cache(tmp_path / "cache", enabled=False)
    with pytest.raises(LiveOfflineError):
        _resolve_source(options, load_config(), cache, DummyProgress())
