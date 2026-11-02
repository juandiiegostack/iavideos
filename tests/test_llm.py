from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cliphunter.config import load_config
from cliphunter.exceptions import LLMError
from cliphunter.llm import LLMClient
from cliphunter.models import CandidateClip, MediaInfo


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self) -> Any:
        return self._payload


def _client() -> LLMClient:
    config = load_config(
        overrides={
            "llm.enabled": True,
            "llm.base_url": "https://llm.example/v1",
            "llm.api_key": "secret-key",
            "llm.model": "test-model",
            "llm.weight": 0.5,
        }
    )
    return LLMClient(config)


def _media() -> MediaInfo:
    return MediaInfo(
        path=Path("video.mp4"),
        duration=3600,
        width=1920,
        height=1080,
        fps=60,
        has_audio=True,
        title="Directo de prueba",
        platform="twitch",
    )


def _clips() -> list[CandidateClip]:
    return [
        CandidateClip(
            index=1,
            start=10,
            end=25,
            peak=12,
            score=0.40,
            title="Momento 1",
            transcript="hola",
        ),
        CandidateClip(
            index=2,
            start=100,
            end=115,
            peak=101,
            score=0.80,
            title="Momento 2",
            transcript="no puede ser",
        ),
    ]


def test_validate_requires_base_url_and_model() -> None:
    config = load_config(overrides={"llm.enabled": True, "llm.base_url": "", "llm.model": ""})
    with pytest.raises(LLMError) as info:
        LLMClient(config).validate()
    assert "CLIPHUNTER_LLM_BASE_URL" in str(info.value)
    assert "CLIPHUNTER_LLM_MODEL" in str(info.value)


def test_endpoint_handles_base_and_full_urls() -> None:
    assert _client()._endpoint() == "https://llm.example/v1/chat/completions"
    config = load_config(
        overrides={
            "llm.base_url": "https://llm.example/v1/chat/completions",
            "llm.model": "m",
        }
    )
    assert LLMClient(config)._endpoint() == "https://llm.example/v1/chat/completions"


def test_rank_parses_scores_titles_and_reorders(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json"]
        content = json.dumps(
            {
                "clips": [
                    {"index": 1, "score": 10, "title": '  "Primero"  ', "reason": "clímax"},
                    {"index": 2, "score": 2, "title": "# Segundo", "reason": "plano"},
                ]
            }
        )
        return FakeResponse(payload={"choices": [{"message": {"content": content}}]})

    import cliphunter.llm as llm_module

    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    clips = _clips()
    ranked = _client().rank(clips, _media())

    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["model"] == "test-model"
    assert "Candidatos" in captured["payload"]["messages"][1]["content"]

    assert [clip.title for clip in ranked] == ["Primero", "Segundo"]
    assert [clip.index for clip in ranked] == [1, 2]
    assert ranked[0].llm_score == 10
    assert ranked[0].llm_reason == "clímax"
    assert ranked[0].score == pytest.approx(0.5 * 0.40 + 0.5 * 1.0)
    assert ranked[1].score == pytest.approx(0.5 * 0.80 + 0.5 * 0.2)


def test_rank_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import cliphunter.llm as llm_module

    monkeypatch.setattr(
        llm_module.requests,
        "post",
        lambda *a, **k: FakeResponse(status_code=401, payload={}, text="unauthorized"),
    )
    with pytest.raises(LLMError) as info:
        _client().rank(_clips(), _media())
    assert "401" in str(info.value)


def test_rank_raises_on_malformed_content(monkeypatch: pytest.MonkeyPatch) -> None:
    import cliphunter.llm as llm_module

    monkeypatch.setattr(
        llm_module.requests,
        "post",
        lambda *a, **k: FakeResponse(payload={"choices": [{"message": {"content": "no soy json"}}]}),
    )
    with pytest.raises(LLMError):
        _client().rank(_clips(), _media())


def test_rank_raises_when_no_candidate_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    import cliphunter.llm as llm_module

    content = json.dumps({"clips": [{"index": 99, "score": 8, "title": "Otro"}]})
    monkeypatch.setattr(
        llm_module.requests,
        "post",
        lambda *a, **k: FakeResponse(payload={"choices": [{"message": {"content": content}}]}),
    )
    with pytest.raises(LLMError):
        _client().rank(_clips(), _media())


def test_parse_content_accepts_markdown_fences_and_lists() -> None:
    assert LLMClient._parse_content('```json\n{"a": 1}\n```') == {"a": 1}
    assert LLMClient._parse_content([{"type": "text", "text": '{"b": 2}'}]) == {"b": 2}
