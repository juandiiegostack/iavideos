"""Topic search over the Whisper transcript (keywords + optional LLM)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..config import Config
from ..models import SignalSeries, TranscriptData
from ..utils import LOG, robust_normalize
from .base import analysis_times, counts_in_windows
from .transcript import _norm_token

# Very common words add noise to topical matching, but are still used as a
# fallback when the query contains nothing else.
_STOPWORDS = {
    "a", "al", "algo", "and", "are", "as", "at", "be", "by", "como", "con",
    "de", "del", "do", "does", "donde", "el", "en", "es", "esta", "este",
    "for", "fue", "from", "ha", "hay", "how", "in", "is", "it", "la", "las",
    "lo", "los", "me", "mi", "my", "no", "o", "of", "on", "or", "para",
    "pero", "por", "que", "se", "ser", "son", "su", "that", "the", "this",
    "to", "un", "una", "unas", "unos", "was", "were", "what", "when",
    "where", "which", "why", "with", "y", "ya",
}


def query_terms(query: str) -> list[str]:
    """Normalize a query into meaningful, unique search terms."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in str(query or "").split():
        token = _norm_token(raw)
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    content = [term for term in normalized if term not in _STOPWORDS and len(term) > 1]
    return content or normalized


def _match_weight(token: str, term: str) -> float:
    if not token or not term:
        return 0.0
    if token == term:
        return 1.0
    # Cheap stemming for plurals/conjugations: "ganga"/"gangas", "nace"/"nacer".
    if len(token) >= 4 and len(term) >= 4 and abs(len(token) - len(term)) <= 3:
        if token.startswith(term) or term.startswith(token):
            return 0.55
    return 0.0


def _keyword_hits(
    transcript: TranscriptData,
    terms: list[str],
    start: float,
    end: float,
) -> tuple[list[tuple[float, str, float]], dict[str, list[float]]]:
    """Find query terms in word-level transcript and build evidence hits."""
    words = sorted(
        (w for w in transcript.words if w.end >= start and w.start <= end),
        key=lambda word: word.start,
    )
    token_list = [_norm_token(word.text) for word in words]
    hits: list[tuple[float, str, float]] = []
    per_term: dict[str, list[float]] = {term: [] for term in terms}

    for word, token in zip(words, token_list):
        best_term = ""
        best_weight = 0.0
        for term in terms:
            weight = _match_weight(token, term)
            if weight > best_weight:
                best_term, best_weight = term, weight
        if best_term:
            hits.append((float(word.start), best_term, best_weight))
            per_term[best_term].append(float(word.start))

    # Bonus when the full multi-word query appears in order.
    ordered_terms = [term for term in terms if term]
    if len(ordered_terms) >= 2:
        sequence = [
            (index, token) for index, token in enumerate(token_list) if token
        ]
        for position in range(len(sequence) - len(ordered_terms) + 1):
            chunk = sequence[position : position + len(ordered_terms)]
            if [token for _, token in chunk] == ordered_terms:
                word_index = chunk[0][0]
                hits.append(
                    (
                        float(words[word_index].start),
                        " ".join(ordered_terms),
                        1.5,
                    )
                )
                break

    hits.sort(key=lambda item: item[0])
    return hits, per_term


def _build_passages(
    transcript: TranscriptData,
    start: float,
    end: float,
    config: Config,
    terms: list[str],
) -> list[dict[str, Any]]:
    """Group words into short passages suitable for an LLM relevance call."""
    words = sorted(
        (w for w in transcript.words if w.end >= start and w.start <= end),
        key=lambda word: word.start,
    )
    if not words:
        return []

    max_seconds = max(5.0, config.f("query.passage_seconds", 30.0))
    pause = max(0.1, config.f("query.pause_threshold", 0.35))
    passages: list[dict[str, Any]] = []
    current: list[Any] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(str(word.text).strip() for word in current).strip()
        if text:
            passages.append(
                {
                    "id": f"p{len(passages) + 1}",
                    "start": float(current[0].start),
                    "end": float(current[-1].end),
                    "text": text[:700],
                }
            )
        current.clear()

    for word in words:
        if current:
            gap = float(word.start) - float(current[-1].end)
            length = float(word.end) - float(current[0].start)
            if gap >= pause or length >= max_seconds:
                flush()
        current.append(word)
    flush()

    max_passages = max(1, config.i("query.llm_max_passages", 40))
    if len(passages) <= max_passages:
        return passages

    def keyword_score(passage: dict[str, Any]) -> float:
        tokens = {_norm_token(token) for token in str(passage["text"]).split()}
        return float(sum(any(_match_weight(token, term) for term in terms) for token in tokens))

    scored = sorted(passages, key=lambda item: -keyword_score(item))
    keep: dict[str, dict[str, Any]] = {
        item["id"]: item for item in scored[: max(1, max_passages // 2)]
    }
    remaining = [item for item in passages if item["id"] not in keep]
    slots = max_passages - len(keep)
    if slots > 0 and remaining:
        step = max(1, len(remaining) // slots)
        for item in remaining[::step][:slots]:
            keep[item["id"]] = item
    return sorted(keep.values(), key=lambda item: float(item["start"]))


def _semantic_series(
    passages: Iterable[dict[str, Any]],
    scores: dict[str, float],
    times: np.ndarray,
    window: float,
) -> np.ndarray:
    semantic = np.zeros(len(times), dtype=np.float32)
    for passage in passages:
        score = scores.get(str(passage.get("id")))
        if score is None:
            continue
        value = float(np.clip(float(score) / 10.0, 0.0, 1.0))
        left = float(passage["start"]) - window / 2.0
        right = float(passage["end"]) + window / 2.0
        mask = (times >= left) & (times <= right)
        semantic[mask] = np.maximum(semantic[mask], value)
    return semantic


def _anchors(
    times: np.ndarray,
    scores: np.ndarray,
    hits: list[tuple[float, str, float]],
    semantic: np.ndarray | None,
    config: Config,
) -> list[tuple[float, str, float]]:
    """Pick local maxima of the query score and attach a readable label."""
    if len(scores) == 0:
        return []
    threshold = max(1e-6, config.f("query.min_score", 0.15))
    local = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] >= scores[index - 1] and scores[index] > scores[index + 1]
    ]
    if not local:
        local = [int(np.argmax(scores))]
    peaks = [index for index in local if float(scores[index]) >= threshold]
    if not peaks and float(scores.max()) >= threshold:
        peaks = [int(np.argmax(scores))]
    peaks.sort(key=lambda index: -float(scores[index]))

    window = config.f("query.window", 20.0)
    anchors: list[tuple[float, str, float]] = []
    merge_distance = config.f("query.merge_distance", 6.0)
    for index in peaks:
        center = float(times[index])
        nearby = [
            (abs(time - center), label, weight)
            for time, label, weight in hits
            if abs(time - center) <= window / 2.0
        ]
        if nearby:
            label = min(nearby, key=lambda item: (-item[2], item[0]))[1]
        elif semantic is not None and float(semantic[index]) > 1e-6:
            label = "relevancia semántica"
        else:
            continue
        if any(abs(center - existing) < merge_distance for existing, _, _ in anchors):
            continue
        anchors.append((center, label, float(scores[index])))
    return anchors[:80]


def analyze_query(
    transcript: TranscriptData | None,
    duration: float,
    config: Config,
    query: str,
    *,
    start: float = 0.0,
    end: float | None = None,
    llm_client: Any | None = None,
) -> SignalSeries:
    """Score transcript passages by keyword overlap and optional LLM relevance."""
    window = config.f("query.window", 20.0)
    hop = config.f("analysis.hop", 2.0)
    range_start = max(0.0, float(start))
    range_end = float(duration) if end is None else min(float(duration), float(end))
    times = analysis_times(max(0.0, range_end - range_start), hop, offset=range_start)
    terms = query_terms(query)

    if transcript is None or not transcript.words:
        return SignalSeries(
            "query", times, np.zeros(len(times), dtype=np.float32), False,
            "búsqueda sin transcripción", detail={"query": query, "terms": terms},
        )
    if not terms:
        return SignalSeries(
            "query", times, np.zeros(len(times), dtype=np.float32), False,
            "consulta vacía", detail={"query": query, "terms": []},
        )

    hits, per_term = _keyword_hits(transcript, terms, range_start, range_end)
    hit_times = np.array([time for time, _, _ in hits], dtype=np.float64)
    hit_weights = np.array([weight for _, _, weight in hits], dtype=np.float64)
    keyword_counts = counts_in_windows(hit_times, times, window, hit_weights)

    coverage_parts: list[np.ndarray] = []
    for term in terms:
        term_times = np.array(per_term.get(term, []), dtype=np.float64)
        if len(term_times) == 0:
            coverage_parts.append(np.zeros(len(times), dtype=np.float32))
        else:
            coverage_parts.append(
                (counts_in_windows(term_times, times, window) > 0).astype(np.float32)
            )
    coverage = (
        np.mean(coverage_parts, axis=0)
        if coverage_parts
        else np.zeros(len(times), dtype=np.float32)
    )
    keyword_score = np.clip(
        0.75 * robust_normalize(keyword_counts) + 0.25 * coverage, 0.0, 1.0
    ).astype(np.float32)

    passages: list[dict[str, Any]] = []
    semantic: np.ndarray | None = None
    if llm_client is not None:
        passages = _build_passages(transcript, range_start, range_end, config, terms)
        if passages:
            try:
                scores = llm_client.score_query(query, passages)
            except Exception as exc:  # noqa: BLE001 - optional signal
                LOG.warning("Relevancia semántica no disponible: %s", exc)
            else:
                if scores:
                    semantic = _semantic_series(passages, scores, times, window)
                if semantic is not None and float(semantic.max()) > 0.0:
                    LOG.info(
                        "Relevancia semántica del LLM: %d fragmentos, máximo %.1f/10.",
                        sum(1 for value in scores.values() if float(value) > 0),
                        float(semantic.max()) * 10.0,
                    )

    if semantic is not None:
        scores = np.clip(0.60 * keyword_score + 0.40 * semantic, 0.0, 1.0).astype(np.float32)
    else:
        scores = keyword_score

    anchors = _anchors(times, scores, hits, semantic, config)
    note = f"{len(hits)} coincidencias"
    if semantic is not None:
        note += " + relevancia LLM"
    if not anchors:
        note = "sin coincidencias para la búsqueda"
    LOG.info("Señal de búsqueda «%s»: %s", query, note)

    return SignalSeries(
        "query",
        times,
        scores,
        True,
        note,
        detail={
            "query": query,
            "terms": terms,
            "hits": hits,
            "semantic": semantic,
            "anchors": anchors,
        },
    )
