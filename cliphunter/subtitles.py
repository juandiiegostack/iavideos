"""Generate ASS subtitles from Whisper word timestamps and burn them in."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .models import TranscriptData, Word
from .utils import LOG, format_ass_time, write_text_atomic


def _escape_ass(text: str) -> str:
    return (
        text.replace("\\", "/")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def _style_header(
    *,
    width: int,
    height: int,
    vertical: bool,
    config: Config,
) -> str:
    font = str(config.get("subtitles.font", "Arial Black"))
    size_key = "subtitles.font_size_vertical" if vertical else "subtitles.font_size_horizontal"
    margin_key = "subtitles.margin_v_vertical" if vertical else "subtitles.margin_v_horizontal"
    size = config.i(size_key, 104 if vertical else 56)
    margin_v = config.i(margin_key, 340 if vertical else 40)
    outline = config.i("subtitles.outline", 7)
    shadow = config.i("subtitles.shadow", 2)
    bold = -1 if config.b("subtitles.bold", True) else 0
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,{bold},0,0,0,100,100,1,0,1,{outline},{shadow},2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _chunk_words(
    words: list[Word],
    *,
    max_words: int,
    max_chars: int,
    max_duration: float,
) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        proposed = current + [word]
        chars = len(" ".join(w.text for w in proposed))
        too_long = len(proposed) > max_words or chars > max_chars
        too_slow = bool(current) and word.start - current[-1].end > 0.9
        too_duration = bool(current) and word.end - current[0].start > max_duration
        if current and (too_long or too_slow or too_duration):
            chunks.append(current)
            current = [word]
        else:
            current = proposed
    if current:
        chunks.append(current)
    return chunks


def build_ass(
    transcript: TranscriptData | None,
    clip_start: float,
    clip_end: float,
    output: str | Path,
    *,
    config: Config,
    vertical: bool,
    width: int,
    height: int,
    mode: str | None = None,
) -> Path | None:
    """Write a word-highlighted ASS file for the selected clip range.

    Returns ``None`` when the clip has no transcribed words, so the caller can
    export without subtitles instead of failing.
    """
    if transcript is None or not transcript.words:
        LOG.warning("No hay palabras transcritas para este clip; se omite el subtítulo.")
        return None

    clip_duration = max(0.05, clip_end - clip_start)
    words = [
        Word(
            start=max(0.0, w.start - clip_start),
            end=min(clip_duration, max(0.0, w.end - clip_start)),
            text=w.text,
            probability=w.probability,
        )
        for w in transcript.words
        if w.end > clip_start and w.start < clip_end
    ]
    words = [w for w in words if w.text.strip() and w.end > w.start]
    if not words:
        LOG.warning("El clip no contiene palabras transcritas; se omite el subtítulo.")
        return None
    words.sort(key=lambda w: (w.start, w.end))

    subtitle_mode = (mode or str(config.get("subtitles.mode", "word"))).lower()
    max_words = max(1, config.i("subtitles.max_words", 4))
    max_chars = max(4, config.i("subtitles.max_chars", 30))
    max_duration = config.f("subtitles.max_duration", 3.0)
    highlight = str(config.get("subtitles.highlight_color", "&H00FFFF&"))

    header = _style_header(width=width, height=height, vertical=vertical, config=config)
    lines: list[str] = []
    chunks = _chunk_words(
        words,
        max_words=max_words,
        max_chars=max_chars,
        max_duration=max_duration,
    )
    for chunk_index, chunk in enumerate(chunks):
        next_chunk_start = (
            chunks[chunk_index + 1][0].start if chunk_index + 1 < len(chunks) else clip_duration
        )
        if subtitle_mode == "chunk":
            start = chunk[0].start
            end = min(clip_duration, next_chunk_start, chunk[-1].end + 0.25)
            text = _escape_ass(" ".join(w.text for w in chunk))
            lines.append(
                f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(max(end, start + 0.2))},"
                f"Default,,0,0,0,,{{\\fad(40,40)}}{text}"
            )
            continue

        # Word mode: one event per word, full chunk visible, current word colored.
        for index, word in enumerate(chunk):
            start = word.start
            if index + 1 < len(chunk):
                next_start = chunk[index + 1].start
            else:
                next_start = min(next_chunk_start, word.end + 0.45)
            end = min(clip_duration, max(start + 0.12, min(next_start, word.end + 0.55)))
            pieces: list[str] = []
            for other_index, other in enumerate(chunk):
                raw = _escape_ass(other.text)
                if other_index == index:
                    pieces.append(f"{{\\1c{highlight}}}{raw}{{\\1c&HFFFFFF&}}")
                else:
                    pieces.append(raw)
            text = " ".join(pieces)
            lines.append(
                f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},"
                f"Default,,0,0,0,,{{\\fad(30,30)}}{text}"
            )

    if not lines:
        return None
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, header + "\n".join(lines) + "\n")
    return target
