from __future__ import annotations

import json
from pathlib import Path

import pytest

from cliphunter.cache import Cache
from cliphunter.config import load_config
from cliphunter.downloader import RemoteSource, SourceSpec
from cliphunter.exceptions import ClipHunterError, SourceError
from cliphunter.models import CandidateClip, MediaInfo
from cliphunter.pipeline import (
    RunOptions,
    _resolve_analysis_range,
    _resolve_source,
    _write_manifest,
)


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


def _media() -> MediaInfo:
    return MediaInfo(
        path=Path("video.mp4"),
        duration=40.0,
        width=640,
        height=360,
        fps=25,
        has_audio=True,
    )


def _entry(start: float, end: float, name: str) -> dict:
    return {
        "index": 1,
        "file": name,
        "subtitle_file": None,
        "start": start,
        "end": end,
        "duration": end - start,
        "peak": 10.0,
        "score": 0.5,
        "signals": {},
        "reasons": [],
        "title": "Titulo",
        "transcript": "",
        "llm_score": None,
        "llm_reason": "",
    }


def test_write_manifest_accumulates_previous_clips(tmp_path: Path) -> None:
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    (output_dir / "clips.json").write_text(
        json.dumps(
            {
                "tool": "ClipHunter",
                "version": "0.1.0",
                "source": {},
                "settings": {},
                "clips": [_entry(5.0, 20.0, "clip_001_old.mp4")],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = _write_manifest(output_dir, _media(), [], load_config(), {}, [])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["clips"]) == 1

    new_clip = CandidateClip(
        index=2,
        start=30.0,
        end=39.0,
        peak=32.0,
        score=0.9,
        title="Nuevo",
        file="clip_002_nuevo.mp4",
    )
    manifest_path = _write_manifest(output_dir, _media(), [new_clip], load_config(), {}, [])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["clips"]) == 2
    assert payload["clips"][0]["file"] == "clip_001_old.mp4"
    assert payload["clips"][1]["file"] == "clip_002_nuevo.mp4"


def test_write_manifest_does_not_duplicate_same_clip_twice(tmp_path: Path) -> None:
    output_dir = tmp_path / "clips"
    output_dir.mkdir()
    new_clip = CandidateClip(
        index=2,
        start=30.0,
        end=39.0,
        peak=32.0,
        score=0.9,
        title="Nuevo",
        file="clip_002_nuevo.mp4",
    )
    _write_manifest(output_dir, _media(), [new_clip], load_config(), {}, [])
    _write_manifest(output_dir, _media(), [new_clip], load_config(), {}, [])
    payload = json.loads((output_dir / "clips.json").read_text(encoding="utf-8"))
    assert len(payload["clips"]) == 1


def test_resolve_analysis_range_rejects_start_not_before_end() -> None:
    with pytest.raises(ClipHunterError) as info:
        _resolve_analysis_range(RunOptions(start=30.0, end=20.0), 120.0)
    assert "--start debe ser menor que --end" in str(info.value)


def test_resolve_analysis_range_rejects_start_out_of_range() -> None:
    with pytest.raises(ClipHunterError) as info:
        _resolve_analysis_range(RunOptions(start=130.0, end=140.0), 120.0)
    assert "está fuera del video" in str(info.value)


def test_resolve_analysis_range_defaults_to_whole_video() -> None:
    assert _resolve_analysis_range(RunOptions(), 120.0) == (0.0, 120.0)


def test_resolve_analysis_range_clamps_end_to_duration() -> None:
    start, end = _resolve_analysis_range(RunOptions(start=10.0, end=999.0), 120.0)
    assert start == 10.0
    assert end == 120.0
