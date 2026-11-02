from __future__ import annotations

from cliphunter.config import load_config
from cliphunter.models import TranscriptData, TranscriptSegment, Word
from cliphunter.signals.transcript import analyze_transcript


def _transcript(words: list[tuple[float, str]]) -> TranscriptData:
    result = []
    for index, (start, text) in enumerate(words):
        result.append(
            Word(start=start, end=start + 0.3, text=text, probability=0.9)
        )
    return TranscriptData(
        words=result,
        segments=[TranscriptSegment(result[0].start, result[-1].end, " ".join(w.text for w in result))],
        language="es",
        duration=120,
        model="test",
    )


def test_keyword_burst_is_detected_and_explained() -> None:
    words: list[tuple[float, str]] = []
    t = 0.0
    while t < 120:
        if 60 <= t <= 62:
            for token in ["no", "puede", "ser", "increíble"]:
                words.append((t, token))
                t += 0.25
        else:
            words.append((t, "hola"))
            t += 1.0
    transcript = _transcript(words)
    config = load_config()
    signal = analyze_transcript(transcript, 120.0, config)
    assert signal.available
    peak_time = float(signal.times[int(signal.values.argmax())])
    assert 55 <= peak_time <= 70
    hits = signal.detail["keyword_hits"]
    labels = {label for _, label, _ in hits}
    assert "no puede ser" in labels
    assert "increíble" in labels


def test_empty_transcript_is_unavailable() -> None:
    config = load_config()
    signal = analyze_transcript(TranscriptData(), 60.0, config)
    assert not signal.available
    assert signal.values.max() == 0.0


def test_excerpt_uses_words_in_clip() -> None:
    transcript = _transcript([(10.0, "no"), (10.5, "puede"), (11.0, "ser")])
    assert transcript.excerpt(9.0, 12.0) == "no puede ser"
    assert transcript.excerpt(0.0, 5.0) == ""
