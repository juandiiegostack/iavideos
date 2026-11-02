from __future__ import annotations

import numpy as np
import pytest

from cliphunter.config import load_config
from cliphunter.models import SignalSeries, TranscriptData, TranscriptSegment, Word
from cliphunter.signals.query import (
    _anchors,
    _build_passages,
    _keyword_hits,
    _semantic_series,
    analyze_query,
    query_terms,
)


def _transcript(words: list[tuple[float, str]], duration: float = 120.0) -> TranscriptData:
    word_objs = [
        Word(start=start, end=start + 0.25, text=text, probability=0.9)
        for start, text in words
    ]
    segments = []
    if word_objs:
        segments = [
            TranscriptSegment(
                word_objs[0].start, word_objs[-1].end, " ".join(w.text for w in word_objs)
            )
        ]
    return TranscriptData(
        words=word_objs,
        segments=segments,
        language="es",
        duration=duration,
        model="test",
    )


def test_query_terms_normalize_and_filter_stopwords() -> None:
    assert query_terms("de dónde nace la ganga") == ["nace", "ganga"]
    assert query_terms("   ") == []
    assert query_terms("") == []
    assert query_terms("no puede ser increíble qué locura") == ["puede", "increible", "locura"]


def test_keyword_hits_match_inside_range_only() -> None:
    transcript = _transcript([(40.0, "hola"), (45.0, "nace"), (50.0, "ganga")])
    hits, per_term = _keyword_hits(transcript, ["ganga"], 0.0, 60.0)
    assert hits == [(50.0, "ganga", 1.0)]
    assert per_term["ganga"] == [50.0]

    hits, _per_term = _keyword_hits(transcript, ["ganga"], 0.0, 49.0)
    assert hits == []


def test_build_passages_splits_on_long_pauses() -> None:
    transcript = _transcript([(0.0, "hola"), (1.0, "mundo"), (10.0, "nace"), (11.0, "ganga")])
    config = load_config()
    passages = _build_passages(transcript, 0.0, 20.0, config, ["nace", "ganga"])
    assert [round(p["start"], 2) for p in passages] == [0.0, 1.0, 10.0, 11.0]
    assert passages[-1]["text"] == "ganga"


def test_semantic_series_maps_passage_scores_to_grid() -> None:
    times = np.arange(0.0, 60.0, 20.0, dtype=np.float32)
    semantic = _semantic_series(
        [{"id": "p1", "start": 0.0, "end": 10.0, "text": "hola"}],
        {"p1": 8.0},
        times,
        20.0,
    )
    assert semantic[0] == pytest.approx(0.8)
    assert semantic[1] == pytest.approx(0.8)


def test_anchors_pick_local_maxima_above_threshold() -> None:
    times = np.arange(0.0, 40.0, 2.0, dtype=np.float32)
    scores = np.zeros(len(times), dtype=np.float32)
    scores[10] = 0.05
    anchors = _anchors(times, scores, [(20.0, "ganga", 1.0)], None, load_config())
    assert anchors == []

    scores[10] = 0.3
    anchors = _anchors(times, scores, [(20.0, "ganga", 1.0)], None, load_config())
    assert anchors == [(20.0, "ganga", pytest.approx(0.3))]


def test_analyze_query_scores_topic_matches() -> None:
    transcript = _transcript([(40.0, "hola"), (45.0, "nace"), (50.0, "ganga")])
    config = load_config()
    signal = analyze_query(transcript, 120.0, config, "de dónde nace la ganga")

    assert signal.name == "query"
    assert signal.available
    assert float(signal.values.max()) > 0.0
    assert signal.detail["terms"] == ["nace", "ganga"]
    assert signal.detail["hits"]
    anchors = signal.detail["anchors"]
    assert anchors
    assert all(0.0 <= anchor[0] <= 120.0 for anchor in anchors)


def test_analyze_query_without_transcript_is_unavailable() -> None:
    signal = analyze_query(None, 60.0, load_config(), "ganga")
    assert not signal.available
    assert signal.values.max() == 0.0
    assert signal.detail["query"] == "ganga"


def test_analyze_query_respects_analysis_range() -> None:
    transcript = _transcript([(40.0, "hola"), (45.0, "nace"), (50.0, "ganga")])

    signal = analyze_query(transcript, 120.0, load_config(), "ganga", start=0.0, end=46.0)
    assert signal.available
    assert float(signal.values.max()) == 0.0
    assert signal.detail["anchors"] == []

    signal = analyze_query(transcript, 120.0, load_config(), "ganga", start=0.0, end=60.0)
    assert signal.available
    assert float(signal.values.max()) > 0.0
    assert all(0.0 <= anchor[0] <= 60.0 for anchor in signal.detail["anchors"])


def test_analyze_query_uses_llm_semantic_relevance() -> None:
    class FakeLLM:
        def score_query(self, query: str, passages: list[dict]) -> dict[str, float]:
            assert query == "nace"
            return {str(passages[0].get("id", "")): 8.0} if passages else {}

    transcript = _transcript([(40.0, "hola"), (45.0, "mundo"), (50.0, "adios")])
    signal = analyze_query(transcript, 120.0, load_config(), "nace", llm_client=FakeLLM())

    assert signal.available
    assert signal.detail["semantic"] is not None
    assert float(signal.values.max()) > 0.0
    assert signal.detail["anchors"]
    assert any("semántica" in str(anchor[1]) for anchor in signal.detail["anchors"])


def test_analyze_query_empty_query_is_unavailable() -> None:
    transcript = _transcript([(10.0, "hola")])
    signal = analyze_query(transcript, 60.0, load_config(), "   ")
    assert not signal.available
    assert signal.detail["terms"] == []
