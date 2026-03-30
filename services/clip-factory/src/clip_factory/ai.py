from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any
from urllib import error, parse, request

from .config import Settings
from .contracts import ClipMetadata, JobState, SegmentCandidate, SourceAsset


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _dedupe_hashtags(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        hashtag = value if value.startswith("#") else f"#{value}"
        hashtag = re.sub(r"[^A-Za-z0-9_#]+", "", hashtag)
        if len(hashtag) < 2:
            continue
        lowered = hashtag.lower()
        if lowered in seen:
            continue
        deduped.append(hashtag)
        seen.add(lowered)
    return deduped


def _strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    def normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized = {key: normalize(value) for key, value in node.items()}
        if normalized.get("type") == "object":
            normalized.setdefault("additionalProperties", False)
        return normalized

    return normalize(schema)


class ExternalAISegmentScorer:
    def __init__(self, endpoint_url: str | None, bearer_token: str | None = None, timeout_seconds: int = 45) -> None:
        self.endpoint_url = endpoint_url
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds

    def is_enabled(self) -> bool:
        return bool(self.endpoint_url)

    def enrich_candidates(
        self,
        job: JobState,
        source: SourceAsset,
        candidates: list[SegmentCandidate],
    ) -> list[SegmentCandidate]:
        if not self.endpoint_url or not candidates:
            return candidates

        payload = {
            "job_id": job.job_id,
            "source": {
                "source_url": source.source_url,
                "title": source.title,
                "description": source.description[:400],
                "content_type": job.input.content_type,
            },
            "candidates": [
                {
                    "segment_id": candidate.segment_id,
                    "start_ms": candidate.start_ms,
                    "end_ms": candidate.end_ms,
                    "score": candidate.score,
                    "hook_text": candidate.hook_text,
                    "keywords": candidate.keywords,
                    "text": candidate.text[:1200],
                }
                for candidate in candidates[:24]
            ],
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        http_request = request.Request(self.endpoint_url, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return candidates

        segments = {item.get("segment_id"): item for item in response_payload.get("segments", []) if item.get("segment_id")}
        updated: list[SegmentCandidate] = []
        for candidate in candidates:
            update = segments.get(candidate.segment_id)
            if not update:
                updated.append(candidate)
                continue
            updated.append(
                replace(
                    candidate,
                    score=float(update.get("score", candidate.score)),
                    reason=str(update.get("reason", candidate.reason)),
                    hook_text=str(update.get("hook_text", candidate.hook_text)),
                    keywords=list(update.get("keywords", candidate.keywords)),
                )
            )
        return sorted(updated, key=lambda item: item.score, reverse=True)


class BaseLLMHybridScorer(ABC):
    def is_enabled(self) -> bool:
        return bool(self._api_key())

    def enrich_candidates(
        self,
        job: JobState,
        source: SourceAsset,
        candidates: list[SegmentCandidate],
    ) -> list[SegmentCandidate]:
        if not self.is_enabled() or not candidates:
            return candidates

        payload = {
            "job_id": job.job_id,
            "content_type": job.input.content_type,
            "source_language": source.language or job.input.language_mode,
            "source": {
                "title": source.title,
                "uploader": source.uploader,
                "description": source.description[:600],
            },
            "candidates": [
                {
                    "segment_id": candidate.segment_id,
                    "start_ms": candidate.start_ms,
                    "end_ms": candidate.end_ms,
                    "heuristic_score": candidate.score,
                    "heuristic_reason": candidate.reason,
                    "confidence": candidate.confidence,
                    "hook_text": candidate.hook_text,
                    "keywords": candidate.keywords[:6],
                    "flags": candidate.flags,
                    "excerpt": candidate.text[:900],
                }
                for candidate in candidates[:18]
            ],
        }
        developer_prompt = (
            "You are a senior short-form video editor for YouTube Shorts and TikTok. "
            "Re-score each candidate clip using the heuristic score as one signal, not as the final answer. "
            "Prefer segments with a clean standalone story, strong opening line, practical value, emotional novelty, and clear payoff. "
            "Penalize intros, outros, sponsorship, vague pronouns, filler, and clips that only make sense with missing visual context. "
            "Keep hook_text in the source language, specific, natural, and at most 90 characters. "
            "Return 3 to 6 useful keywords for subtitle highlighting. "
            "You must return one result for every input segment_id."
        )
        schema = {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_id": {"type": "string"},
                            "score": {"type": "number"},
                            "reason": {"type": "string"},
                            "hook_text": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["segment_id", "score", "reason", "hook_text", "keywords"],
                    },
                }
            },
            "required": ["segments"],
        }
        response_payload = self._generate_json(
            developer_prompt=developer_prompt,
            user_payload=payload,
            schema_name="clip_candidate_ranking",
            schema_description="Re-scored short-video clip candidates.",
            schema=schema,
            max_output_tokens=3000,
        )
        if not response_payload:
            return candidates

        updates = {item.get("segment_id"): item for item in response_payload.get("segments", []) if item.get("segment_id")}
        enriched: list[SegmentCandidate] = []
        for candidate in candidates:
            update = updates.get(candidate.segment_id)
            if not update:
                enriched.append(candidate)
                continue

            llm_score = _clamp_score(float(update.get("score", candidate.score)))
            blended_score = round((candidate.score * 0.45) + (llm_score * 0.55), 2)
            llm_reason = str(update.get("reason", candidate.reason)).strip()[:180]
            llm_hook = str(update.get("hook_text", candidate.hook_text)).strip()[:90] or candidate.hook_text
            llm_keywords = [str(item).strip().lower() for item in update.get("keywords", []) if str(item).strip()]
            enriched.append(
                replace(
                    candidate,
                    score=blended_score,
                    reason=llm_reason or candidate.reason,
                    hook_text=llm_hook,
                    keywords=llm_keywords[:6] or candidate.keywords,
                )
            )
        return sorted(enriched, key=lambda item: item.score, reverse=True)

    def enrich_clip_metadata(
        self,
        job: JobState,
        source: SourceAsset,
        fallback_by_clip_id: dict[str, tuple[SegmentCandidate, ClipMetadata]],
    ) -> dict[str, ClipMetadata]:
        if not self.is_enabled() or not fallback_by_clip_id:
            return {clip_id: item[1] for clip_id, item in fallback_by_clip_id.items()}

        payload = {
            "job_id": job.job_id,
            "content_type": job.input.content_type,
            "source_language": source.language or job.input.language_mode,
            "source_title": source.title,
            "source_uploader": source.uploader,
            "clips": [
                {
                    "clip_id": clip_id,
                    "segment_id": segment.segment_id,
                    "start_ms": segment.start_ms,
                    "end_ms": segment.end_ms,
                    "score": segment.score,
                    "reason": segment.reason,
                    "hook_text": segment.hook_text,
                    "keywords": segment.keywords[:6],
                    "excerpt": segment.text[:900],
                    "fallback_titles": fallback.titles[:3],
                    "fallback_caption": fallback.caption,
                    "fallback_hashtags": fallback.hashtags[:12],
                }
                for clip_id, (segment, fallback) in fallback_by_clip_id.items()
            ],
        }
        developer_prompt = (
            "You are a social video copywriter. Rewrite the metadata for each selected clip in the source language. "
            "Create punchy but truthful titles, a clean caption, and focused highlight keywords. "
            "Avoid generic patterns like 'in under a minute'. "
            "Titles should feel clickable but not spammy, hashtags should be relevant, and selection_reason should explain why the clip is strong."
        )
        schema = {
            "type": "object",
            "properties": {
                "clips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "clip_id": {"type": "string"},
                            "titles": {"type": "array", "items": {"type": "string"}},
                            "caption": {"type": "string"},
                            "hashtags": {"type": "array", "items": {"type": "string"}},
                            "highlight_keywords": {"type": "array", "items": {"type": "string"}},
                            "selection_reason": {"type": "string"},
                        },
                        "required": ["clip_id", "titles", "caption", "hashtags", "highlight_keywords", "selection_reason"],
                    },
                }
            },
            "required": ["clips"],
        }
        response_payload = self._generate_json(
            developer_prompt=developer_prompt,
            user_payload=payload,
            schema_name="clip_metadata_rewrite",
            schema_description="Upload-ready metadata for selected short-video clips.",
            schema=schema,
            max_output_tokens=2800,
        )
        if not response_payload:
            return {clip_id: item[1] for clip_id, item in fallback_by_clip_id.items()}

        metadata_by_clip_id: dict[str, ClipMetadata] = {}
        rewrites = {item.get("clip_id"): item for item in response_payload.get("clips", []) if item.get("clip_id")}
        for clip_id, (segment, fallback) in fallback_by_clip_id.items():
            update = rewrites.get(clip_id)
            if not update:
                metadata_by_clip_id[clip_id] = fallback
                continue

            titles = [str(item).strip()[:90] for item in update.get("titles", []) if str(item).strip()]
            caption = str(update.get("caption", fallback.caption)).strip()[:220] or fallback.caption
            hashtags = _dedupe_hashtags([str(item).strip() for item in update.get("hashtags", [])])[:15]
            highlight_keywords = [str(item).strip().lower() for item in update.get("highlight_keywords", []) if str(item).strip()]
            selection_reason = str(update.get("selection_reason", fallback.selection_reason)).strip()[:220] or fallback.selection_reason

            metadata_by_clip_id[clip_id] = ClipMetadata(
                titles=(titles[:3] or fallback.titles[:3]),
                caption=caption,
                hashtags=(hashtags or fallback.hashtags[:15]),
                highlight_keywords=(highlight_keywords[:6] or fallback.highlight_keywords[:6]),
                source_timestamp_label=fallback.source_timestamp_label,
                selection_reason=selection_reason,
            )
        return metadata_by_clip_id

    @abstractmethod
    def _api_key(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def _generate_json(
        self,
        developer_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema_description: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any] | None:
        raise NotImplementedError


class OpenAIHybridScorer(BaseLLMHybridScorer):
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.base_url = settings.openai_base_url
        self.timeout_seconds = settings.openai_timeout_seconds
        self.reasoning_effort = settings.openai_reasoning_effort

    def _api_key(self) -> str | None:
        return self.api_key

    def _generate_json(
        self,
        developer_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema_description: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        body: dict[str, Any] = {
            "model": self.model,
            "instructions": developer_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)}],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "description": schema_description,
                    "strict": True,
                    "schema": _strict_json_schema(schema),
                }
            },
            "max_output_tokens": max_output_tokens,
        }
        if self.reasoning_effort and self._supports_reasoning():
            body["reasoning"] = {"effort": self.reasoning_effort}

        http_request = request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        direct = response_payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            try:
                return json.loads(direct)
            except json.JSONDecodeError:
                return None

        for item in response_payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        continue
        return None

    def _supports_reasoning(self) -> bool:
        model = self.model.lower()
        return model.startswith("gpt-5") or model.startswith("o")


class GeminiHybridScorer(BaseLLMHybridScorer):
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.base_url = settings.gemini_base_url.rstrip("/")
        self.timeout_seconds = settings.gemini_timeout_seconds

    def _api_key(self) -> str | None:
        return self.api_key

    def _generate_json(
        self,
        developer_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema_description: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any] | None:
        if not self.api_key:
            return None

        endpoint = f"{self.base_url}/models/{self.model}:generateContent?key={parse.quote(self.api_key)}"
        body = {
            "systemInstruction": {
                "parts": [{"text": developer_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": json.dumps(user_payload, ensure_ascii=False)}],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "maxOutputTokens": max_output_tokens,
                "temperature": 0.2,
            },
        }
        http_request = request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        for candidate in response_payload.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue
            for part in content.get("parts", []):
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        continue
        return None


class HybridAISegmentScorer:
    def __init__(self, settings: Settings) -> None:
        self.external = ExternalAISegmentScorer(settings.ai_scorer_url, settings.ai_scorer_bearer_token)
        self.openai = OpenAIHybridScorer(settings)
        self.gemini = GeminiHybridScorer(settings)
        self.llm_provider = settings.llm_provider

    def _active_llm(self) -> BaseLLMHybridScorer | None:
        if self.llm_provider == "openai":
            return self.openai if self.openai.is_enabled() else None
        if self.llm_provider == "gemini":
            return self.gemini if self.gemini.is_enabled() else None
        if self.openai.is_enabled():
            return self.openai
        if self.gemini.is_enabled():
            return self.gemini
        return None

    def enrich_candidates(self, job: JobState, source: SourceAsset, candidates: list[SegmentCandidate]) -> list[SegmentCandidate]:
        enriched = candidates
        if self.external.is_enabled():
            enriched = self.external.enrich_candidates(job, source, enriched)
        llm = self._active_llm()
        if llm:
            enriched = llm.enrich_candidates(job, source, enriched)
        return sorted(enriched, key=lambda item: item.score, reverse=True)

    def enrich_clip_metadata(
        self,
        job: JobState,
        source: SourceAsset,
        fallback_by_clip_id: dict[str, tuple[SegmentCandidate, ClipMetadata]],
    ) -> dict[str, ClipMetadata]:
        llm = self._active_llm()
        if llm:
            return llm.enrich_clip_metadata(job, source, fallback_by_clip_id)
        return {clip_id: item[1] for clip_id, item in fallback_by_clip_id.items()}
