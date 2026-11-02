from __future__ import annotations

from pathlib import Path

from cliphunter.cache import Cache, source_key


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "cache")
    key = "youtube_abc123"
    cache.save_json(key, "chat.json", {"messages": [1, 2, 3]})
    assert cache.exists(key, "chat.json")
    assert cache.load_json(key, "chat.json") == {"messages": [1, 2, 3]}
    cache.clear(key)
    assert not cache.exists(key, "chat.json")


def test_local_source_key_is_stable_and_specific(tmp_path: Path) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"x" * 100)
    second.write_bytes(b"x" * 100)
    key_a = source_key(local_path=first)
    key_a_again = source_key(local_path=first)
    key_b = source_key(local_path=second)
    assert key_a == key_a_again
    assert key_a != key_b
    assert key_a.startswith("local_a_")


def test_remote_key_uses_platform_and_id() -> None:
    key = source_key(platform="Twitch", video_id="123456", url="https://twitch.tv/videos/123456")
    assert key.startswith("twitch_123456")


def test_disabled_cache_uses_temporary_directory(tmp_path: Path) -> None:
    cache = Cache(tmp_path / "unused", enabled=False)
    assert cache.enabled is False
    assert cache.root.exists()
    cache.save_json("x", "y.json", {"ok": True})
    assert cache.load_json("x", "y.json") == {"ok": True}
