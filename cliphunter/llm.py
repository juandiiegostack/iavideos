"""Optional OpenAI-compatible LLM ranking and title generation."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import Config
from .exceptions import LLMError
from .models import CandidateClip, MediaInfo
from .utils import LOG, format_hms


class LLMClient:
    """Minimal client for any OpenAI-compatible ``/chat/completions`` API."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.base_url = str(config.get("llm.base_url", "") or "").strip()
        self.api_key = str(config.get("llm.api_key", "") or "").strip()
        self.model = str(config.get("llm.model", "") or "").strip()
        self.timeout = config.f("llm.timeout", 60.0)
        self.temperature = config.f("llm.temperature", 0.4)
        self.weight = config.f("llm.weight", 0.30)

    def validate(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("CLIPHUNTER_LLM_BASE_URL")
        if not self.model:
            missing.append("CLIPHUNTER_LLM_MODEL")
        if missing:
            raise LLMError(
                "Falta configuración del LLM: "
                + ", ".join(missing)
                + ".\nDefine esas variables de entorno (o en .env), por ejemplo:\n"
                "  CLIPHUNTER_LLM_BASE_URL=https://api.openai.com/v1\n"
                "  CLIPHUNTER_LLM_API_KEY=...\n"
                "  CLIPHUNTER_LLM_MODEL=gpt-4o-mini"
            )

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _prompt(self, clips: list[CandidateClip], media: MediaInfo) -> list[dict[str, str]]:
        rows = []
        for clip in clips:
            rows.append(
                {
                    "index": clip.index,
                    "rango": f"{format_hms(clip.start)}-{format_hms(clip.end)}",
                    "puntaje_heuristico": round(clip.score, 3),
                    "senales": {k: round(v, 2) for k, v in clip.breakdown.items()},
                    "motivos": clip.reasons,
                    "transcripcion": clip.transcript or "(sin voz detectada)",
                }
            )
        system = (
            "Eres un editor experto de clips para TikTok, YouTube Shorts y Reels. "
            "Evalúa qué tan 'clipeable' es cada momento de un directo. "
            "Busca risas, sorpresa, tensión, clímax, reacciones fuertes o frases memorables. "
            "Penaliza momentos planos, confusos o sin gancho. "
            "Devuelve SOLO JSON válido, sin markdown, con esta forma exacta: "
            '{"clips":[{"index":1,"score":8,"title":"Título corto sin hashtags",'
            '"reason":"una frase"}]}. '
            "score es un entero de 1 a 10. title debe tener menos de 60 caracteres, "
            "estar en español y no incluir comillas raras."
        )
        user = (
            f"Fuente: {media.title or media.platform}. Duración: {format_hms(media.duration)}.\n"
            f"Candidatos:\n{json.dumps(rows, ensure_ascii=False, indent=1)}"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _parse_content(content: Any) -> dict[str, Any]:
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        text = str(content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("El LLM no devolvió un objeto JSON.")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"JSON inválido del LLM: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("El LLM devolvió un JSON que no es un objeto.")
        return data

    def rank(self, clips: list[CandidateClip], media: MediaInfo) -> list[CandidateClip]:
        """Score and title clips, then reorder them by the blended score."""
        self.validate()
        if not clips:
            return clips
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": self._prompt(clips, media),
            "temperature": self.temperature,
            "max_tokens": 1600,
        }
        LOG.info("Consultando al LLM (%s) para %d clips…", self.model, len(clips))
        try:
            response = requests.post(
                self._endpoint(), headers=headers, json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise LLMError(f"No se pudo conectar con el LLM: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"El LLM respondió HTTP {response.status_code}: {response.text[:400]}"
            )
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Respuesta inesperada del LLM: {response.text[:400]}") from exc

        parsed = self._parse_content(content)
        entries = parsed.get("clips") or parsed.get("results") or []
        if not isinstance(entries, list):
            raise LLMError("El JSON del LLM no contiene una lista 'clips'.")
        by_index: dict[int, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            by_index[index] = entry

        matched = 0
        for clip in clips:
            entry = by_index.get(clip.index)
            if not entry:
                continue
            try:
                llm_score = float(entry.get("score", 0))
            except (TypeError, ValueError):
                continue
            llm_score = max(1.0, min(10.0, llm_score))
            title = str(entry.get("title", "")).strip().strip('"“”')
            title = re.sub(r"^#+\s*", "", title)
            clip.llm_score = llm_score
            clip.llm_reason = str(entry.get("reason", "")).strip()[:200]
            if title:
                clip.title = title[:60]
            clip.score = (1.0 - self.weight) * clip.score + self.weight * (llm_score / 10.0)
            matched += 1
        if not matched:
            raise LLMError("El LLM no devolvió puntajes para ninguno de los candidatos.")

        clips.sort(key=lambda c: -c.score)
        for index, clip in enumerate(clips, start=1):
            clip.index = index
        LOG.info("LLM aplicado a %d clips.", matched)
        return clips

    def score_query(
        self, query: str, passages: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Score transcript passages by semantic relevance to ``query`` (0-10)."""
        self.validate()
        if not passages:
            return {}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        batch_size = max(1, self.config.i("query.llm_batch", 10))
        system = (
            "Eres un buscador semántico de clips de directos. "
            "Puntúa del 0 al 10 cuánto habla cada fragmento del tema solicitado. "
            "Ten en cuenta sinónimos, paráfrasis y contexto, no solo palabras exactas. "
            "0 = no tiene nada que ver; 10 = habla directamente del tema. "
            "Devuelve SOLO JSON válido, sin markdown, con esta forma exacta: "
            '{"results":[{"id":"p1","score":8}]}.'
        )

        scores: dict[str, float] = {}
        for offset in range(0, len(passages), batch_size):
            batch = passages[offset : offset + batch_size]
            rows = [
                {
                    "id": str(passage.get("id", "")),
                    "inicio": round(float(passage.get("start", 0.0)), 1),
                    "fin": round(float(passage.get("end", 0.0)), 1),
                    "texto": str(passage.get("text", ""))[:700],
                }
                for passage in batch
            ]
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": (
                            f"Tema buscado: {query}\n"
                            f"Fragmentos:\n{json.dumps(rows, ensure_ascii=False, indent=1)}"
                        ),
                    },
                ],
                "temperature": min(self.temperature, 0.2),
                "max_tokens": 1200,
            }
            LOG.info(
                "Consultando al LLM (%s) por relevancia de %d fragmentos…",
                self.model,
                len(batch),
            )
            try:
                response = requests.post(
                    self._endpoint(), headers=headers, json=payload, timeout=self.timeout
                )
            except requests.RequestException as exc:
                raise LLMError(f"No se pudo conectar con el LLM: {exc}") from exc
            if response.status_code >= 400:
                raise LLMError(
                    f"El LLM respondió HTTP {response.status_code}: {response.text[:400]}"
                )
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError(
                    f"Respuesta inesperada del LLM: {response.text[:400]}"
                ) from exc

            parsed = self._parse_content(content)
            entries = parsed.get("results") or parsed.get("passages") or []
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                passage_id = str(entry.get("id", "")).strip()
                if not passage_id:
                    continue
                try:
                    value = float(entry.get("score", 0))
                except (TypeError, ValueError):
                    continue
                scores[passage_id] = max(0.0, min(10.0, value))
        return scores
