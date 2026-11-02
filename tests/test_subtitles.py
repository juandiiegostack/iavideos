from __future__ import annotations

from pathlib import Path

from cliphunter.config import load_config
from cliphunter.models import TranscriptData, TranscriptSegment, Word
from cliphunter.subtitles import build_ass


def _transcript() -> TranscriptData:
    tokens = [
        (10.0, 10.4, "no"),
        (10.4, 10.8, "puede"),
        (10.8, 11.2, "ser,"),
        (11.3, 11.7, "increíble"),
        (15.0, 15.4, "vamos"),
    ]
    words = [Word(start=s, end=e, text=t, probability=0.9) for s, e, t in tokens]
    return TranscriptData(
        words=words,
        segments=[TranscriptSegment(10.0, 15.4, "no puede ser, increíble vamos")],
        language="es",
        duration=30.0,
        model="test",
    )


def test_build_ass_word_mode_highlights_each_word(tmp_path: Path) -> None:
    config = load_config()
    output = tmp_path / "clip.ass"
    result = build_ass(
        _transcript(),
        10.0,
        16.0,
        output,
        config=config,
        vertical=True,
        width=1080,
        height=1920,
    )
    assert result == output
    text = output.read_text(encoding="utf-8")
    assert "[Script Info]" in text
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "\\1c&H00FFFF&" in text
    # Five words produce at least five highlighted events.
    assert text.count("Dialogue:") >= 5
    assert "increíble" in text


def test_build_ass_chunk_mode_uses_one_line_per_chunk(tmp_path: Path) -> None:
    config = load_config(overrides={"subtitles.mode": "chunk", "subtitles.max_words": 2})
    output = tmp_path / "clip.ass"
    result = build_ass(
        _transcript(),
        10.0,
        16.0,
        output,
        config=config,
        vertical=False,
        width=1280,
        height=720,
    )
    assert result == output
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")]
    assert len(lines) == 3
    assert "PlayResX: 1280" in output.read_text(encoding="utf-8")


def test_build_ass_returns_none_without_words_in_range(tmp_path: Path) -> None:
    config = load_config()
    output = tmp_path / "empty.ass"
    assert build_ass(None, 0.0, 5.0, output, config=config, vertical=False, width=1280, height=720) is None
    assert build_ass(_transcript(), 0.0, 5.0, output, config=config, vertical=False, width=1280, height=720) is None
    assert not output.exists()


def test_ass_times_are_relative_to_clip_start(tmp_path: Path) -> None:
    config = load_config(overrides={"subtitles.mode": "chunk", "subtitles.max_words": 10})
    output = tmp_path / "clip.ass"
    build_ass(
        _transcript(),
        10.0,
        16.0,
        output,
        config=config,
        vertical=False,
        width=1280,
        height=720,
    )
    dialogue = [line for line in output.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")][0]
    assert ",0:00:00.00," in dialogue
