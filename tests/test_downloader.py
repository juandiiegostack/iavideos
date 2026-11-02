from __future__ import annotations

from pathlib import Path

import pytest

from cliphunter.downloader import (
    _offline_message,
    classify_url,
    find_media_file,
    local_source,
)
from cliphunter.exceptions import SourceError, UnsupportedSourceError


@pytest.mark.parametrize(
    ("url", "platform", "kind", "video_id", "channel"),
    [
        ("https://www.twitch.tv/videos/123456789", "twitch", "vod", "123456789", ""),
        ("https://twitch.tv/videos/123456789?t=1h2m", "twitch", "vod", "123456789", ""),
        ("https://www.twitch.tv/ibai", "twitch", "live", "", "ibai"),
        ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "youtube", "vod", "jNQXAC9IVRw", ""),
        ("https://youtu.be/jNQXAC9IVRw", "youtube", "vod", "jNQXAC9IVRw", ""),
        ("https://www.youtube.com/shorts/jNQXAC9IVRw", "youtube", "vod", "jNQXAC9IVRw", ""),
        ("https://www.youtube.com/@channel/live", "youtube", "live", "", "channel"),
        ("https://kick.com/video/12345678-abcd-1234-abcd-1234567890ab", "kick", "vod", "12345678-abcd-1234-abcd-1234567890ab", ""),
        ("https://kick.com/xqc", "kick", "live", "", "xqc"),
        ("https://example.com/media/clip.mp4", "direct", "vod", "", ""),
        ("https://example.com/live/stream.m3u8", "direct", "live", "", ""),
    ],
)
def test_classify_supported_urls(
    url: str, platform: str, kind: str, video_id: str, channel: str
) -> None:
    spec = classify_url(url)
    assert (spec.platform, spec.kind, spec.video_id, spec.channel) == (
        platform,
        kind,
        video_id,
        channel,
    )
    assert spec.url == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://twitch.tv/videos/1",
        "https://twitch.tv/user/clips/FunnyClip",
        "https://youtube.com/feed/subscriptions",
        "https://kick.com/categories",
        "https://vimeo.com/12345",
    ],
)
def test_classify_rejects_unsupported_urls(url: str) -> None:
    with pytest.raises((UnsupportedSourceError, SourceError)):
        classify_url(url)


def test_classify_accepts_url_without_scheme() -> None:
    spec = classify_url("twitch.tv/videos/42")
    assert spec.platform == "twitch"
    assert spec.video_id == "42"


def test_find_media_file_prefers_mp4(tmp_path: Path) -> None:
    (tmp_path / "source.mkv").write_bytes(b"x" * 10)
    (tmp_path / "source.mp4").write_bytes(b"x" * 5)
    (tmp_path / "source.info.json").write_text("{}", encoding="utf-8")
    (tmp_path / "other.mp4").write_bytes(b"x" * 999)
    found = find_media_file(tmp_path)
    assert found is not None
    assert found.name == "source.mp4"


def test_find_media_file_returns_none_when_empty(tmp_path: Path) -> None:
    assert find_media_file(tmp_path) is None


def test_local_source_validates_file(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        local_source(tmp_path / "missing.mp4")

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(SourceError):
        local_source(empty)


def test_offline_message_detection() -> None:
    assert _offline_message("ERROR: The channel is not currently live")
    assert _offline_message("No playable streams found")
    assert not _offline_message("HTTP Error 403: Forbidden")
