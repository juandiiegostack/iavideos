from __future__ import annotations

import json
from pathlib import Path

import pytest

from cliphunter.config import load_config
from cliphunter.downloader import _load_chat_file, _normalize_comment
from cliphunter.models import ChatMessage
from cliphunter.signals.chat import analyze_chat


def test_chat_burst_is_detected() -> None:
    messages: list[ChatMessage] = []
    t = 0.0
    while t < 120:
        if 60 <= t <= 70:
            for _ in range(10):
                messages.append(ChatMessage(t, "KEKW no way insane", ["KEKW"]))
                t += 0.1
        else:
            messages.append(ChatMessage(t, "hola", []))
            t += 2.0

    config = load_config(overrides={"chat.min_messages": 20})
    signal = analyze_chat(messages, 120.0, config)
    assert signal.available
    assert len(signal.values) == len(signal.times)
    peak_time = float(signal.times[int(signal.values.argmax())])
    assert 55 <= peak_time <= 75
    assert float(signal.values.max()) > 0.8
    assert float(signal.detail["rate"].max()) > 2.0
    assert float(signal.detail["emote_density"].max()) > 0.3


def test_chat_with_too_few_messages_is_unavailable() -> None:
    config = load_config()
    signal = analyze_chat([ChatMessage(1.0, "hi"), ChatMessage(2.0, "hello")], 60.0, config)
    assert not signal.available
    assert signal.values.max() == 0.0


def test_no_chat_is_unavailable() -> None:
    config = load_config()
    signal = analyze_chat(None, 60.0, config)
    assert not signal.available
    assert len(signal.values) == len(signal.times)


def test_normalize_comment_supports_compact_and_twitch_shapes() -> None:
    compact = _normalize_comment({"t": 12.5, "text": "KEKW", "emotes": ["KEKW"]})
    assert compact is not None
    assert compact.t == 12.5
    assert compact.emotes == ["KEKW"]

    twitch = _normalize_comment(
        {
            "content_offset_seconds": 3,
            "message": {"body": "PogChamp", "emotes": [{"name": "PogChamp"}]},
            "author": {"name": "viewer"},
        }
    )
    assert twitch is not None
    assert twitch.t == 3.0
    assert twitch.author == "viewer"
    assert twitch.emotes == ["PogChamp"]


def test_load_twitchdownloader_fixture() -> None:
    path = Path(__file__).parent / "fixtures" / "chat_twitchdownloader.json"
    messages = _load_chat_file(path)
    assert len(messages) == 3
    assert messages[0].t == 12.5
    assert "KEKW" in messages[0].text


def test_load_jsonl_and_compact_json(tmp_path: Path) -> None:
    jsonl = tmp_path / "chat.jsonl"
    jsonl.write_text(
        '{"t": 1, "text": "hola"}\n{"t": 2, "text": "gg"}\n', encoding="utf-8"
    )
    messages = _load_chat_file(jsonl)
    assert [m.t for m in messages] == [1.0, 2.0]

    compact = tmp_path / "chat.json"
    compact.write_text(json.dumps([{"t": 4, "text": "LUL"}]), encoding="utf-8")
    assert _load_chat_file(compact)[0].text == "LUL"


def test_invalid_chat_path_raises(tmp_path: Path) -> None:
    from cliphunter.exceptions import ChatUnavailable

    with pytest.raises(ChatUnavailable):
        _load_chat_file(tmp_path / "missing.json")
