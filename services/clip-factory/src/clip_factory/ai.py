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


def _resolve_output_language(job: JobState, source: SourceAsset) -> str:
    requested = (job.input.language_mode or "").strip().lower()
    source_language = (source.language or "").strip().lower()
    if requested and requested not in {"same_as_source", "source", "auto"}:
        return requested
    return source_language or requested


def _language_instruction(job: JobState, source: SourceAsset) -> str:
    language = _resolve_output_language(job, source)
    if language.startswith("en") or language in {"english"}:
        return (
            "Write everything in English only. Do not translate into Indonesian or mix in Indonesian phrasing. "
            "Keep wording natural for native English-speaking viewers."
        )
    if language.startswith("id") or language in {"indonesian", "bahasa indonesia", "bahasa"}:
        return (
            "Write everything in Indonesian only. Do not translate into English except for product names or terms that are naturally said in English. "
            "Keep wording natural for Indonesian viewers."
        )
    return "Write everything in the same language as the source transcript. Do not translate into another language."


def _news_candidate_instruction() -> str:
    return (
        "For news clips, strongly prefer segments that surface the headline, what changed, who said it, why it matters now, and the immediate consequence. "
        "Prefer crisp escalation, clear cause-and-effect, and standalone context within the first sentence. "
        "Penalize analysis-heavy tangents, niche tactical detail without payoff, and segments that require too much prior context. "
        "Do not sensationalize or invent implications beyond what the source supports."
    )


def _news_metadata_instruction() -> str:
    return (
        "For news clips, write titles and hook text like sharp, accurate news headlines. "
        "Lead with the update, escalation, or consequence. "
        "Keep captions concise and factual: what happened, what changed, and why it matters now. "
        "Avoid creator-style filler, vague curiosity bait, or hype that is not grounded in the source."
    )


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
            "You are a senior short-form video producer optimizing clips for YouTube Shorts and TikTok. "
            "Re-score each candidate clip using the heuristic score as one signal, not as the final answer. "
            "Prefer segments with a strong first sentence, standalone context, practical takeaway, emotional novelty, and a clean payoff in under 60 seconds. "
            "Penalize intros, outros, sponsorship, vague pronouns, filler, rambling, and clips that depend on missing on-screen context. "
            f"{_language_instruction(job, source)} "
            "Write hook_text in the source language using natural spoken style. If the speaker sounds casual, preserve that tone naturally. "
            "Do not write generic teaser copy, fake urgency, or language mixing that is not present in the source. "
            "Keep hook_text concrete, punchy, and at most 90 characters. "
            "Return 3 to 6 useful highlight keywords for subtitles; avoid stopwords and generic words. "
            "Reason should be a concise editorial explanation of why the clip is strong or weak. "
            f"{_news_candidate_instruction() if job.input.content_type == 'news' else ''} "
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
            "You are a top-tier social video copywriter. Rewrite the metadata for each selected clip in the source language. "
            f"{_language_instruction(job, source)} "
            "Create a punchy hook card, 3 strong title options, a clean caption, relevant hashtags, focused highlight keywords, and one concise editorial selection_reason. "
            "Avoid generic patterns like 'in under a minute', avoid robotic phrasing, and avoid hashtags built from stopwords. "
            "If the source speaker sounds casual, preserve that tone naturally. Do not invent facts. "
            "Hook text should be short, specific, and scroll-stopping without sounding clickbait. "
            "Titles should be clickable but truthful. Caption should summarize the payoff in 1-2 natural sentences. "
            f"{_news_metadata_instruction() if job.input.content_type == 'news' else ''} "
            "selection_reason should explain why people would keep watching or replay the clip."
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
                            "hook_text": {"type": "string"},
                            "titles": {"type": "array", "items": {"type": "string"}},
                            "caption": {"type": "string"},
                            "hashtags": {"type": "array", "items": {"type": "string"}},
                            "highlight_keywords": {"type": "array", "items": {"type": "string"}},
                            "selection_reason": {"type": "string"},
                        },
                        "required": ["clip_id", "hook_text", "titles", "caption", "hashtags", "highlight_keywords", "selection_reason"],
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

            hook_text = str(update.get("hook_text", fallback.hook_text)).strip()[:90] or fallback.hook_text
            titles = [str(item).strip()[:90] for item in update.get("titles", []) if str(item).strip()]
            caption = str(update.get("caption", fallback.caption)).strip()[:220] or fallback.caption
            hashtags = _dedupe_hashtags([str(item).strip() for item in update.get("hashtags", [])])[:15]
            highlight_keywords = [str(item).strip().lower() for item in update.get("highlight_keywords", []) if str(item).strip()]
            selection_reason = str(update.get("selection_reason", fallback.selection_reason)).strip()[:220] or fallback.selection_reason

            metadata_by_clip_id[clip_id] = ClipMetadata(
                hook_text=hook_text,
                titles=(titles[:3] or fallback.titles[:3]),
                caption=caption,
                hashtags=(hashtags or fallback.hashtags[:15]),
                highlight_keywords=(highlight_keywords[:6] or fallback.highlight_keywords[:6]),
                source_timestamp_label=fallback.source_timestamp_label,
                selection_reason=selection_reason,
            )
        return metadata_by_clip_id

    def rewrite_subtitle_chunks(
        self,
        job: JobState,
        source: SourceAsset,
        segment: SegmentCandidate,
        chunks: list[str],
    ) -> list[str] | None:
        if not self.is_enabled() or not chunks:
            return chunks

        payload = {
            "job_id": job.job_id,
            "content_type": job.input.content_type,
            "source_language": source.language or job.input.language_mode,
            "source_title": source.title,
            "clip_context": {
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "heuristic_hook_text": segment.hook_text,
                "excerpt": segment.text[:1200],
            },
            "subtitle_chunks": [
                {
                    "chunk_id": f"chunk_{index+1:02d}",
                    "text": text,
                }
                for index, text in enumerate(chunks)
            ],
        }
        developer_prompt = (
            "You are a subtitle editor for short-form video. "
            "Clean ASR mistakes so subtitles match the spoken audio more closely, but do not invent facts. "
            f"{_language_instruction(job, source)} "
            "Keep the original speaker tone. Remove obvious mistranscriptions, repeated filler, broken words, and awkward phrasing. "
            "Keep each subtitle chunk compact, readable on mobile, and semantically equivalent to what was said. "
            "Do not translate to another language unless the source chunk is already mixing languages naturally. "
            "Return one cleaned subtitle line for every chunk_id."
        )
        schema = {
            "type": "object",
            "properties": {
                "subtitle_chunks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["chunk_id", "text"],
                    },
                }
            },
            "required": ["subtitle_chunks"],
        }
        response_payload = self._generate_json(
            developer_prompt=developer_prompt,
            user_payload=payload,
            schema_name="subtitle_chunk_cleanup",
            schema_description="Cleaned subtitle chunks aligned to the original spoken audio.",
            schema=schema,
            max_output_tokens=2200,
        )
        if not response_payload:
            return chunks

        rewritten_by_id = {
            item.get("chunk_id"): str(item.get("text", "")).strip()
            for item in response_payload.get("subtitle_chunks", [])
            if item.get("chunk_id")
        }
        rewritten_chunks: list[str] = []
        for index, original in enumerate(chunks):
            chunk_id = f"chunk_{index+1:02d}"
            rewritten = rewritten_by_id.get(chunk_id) or original
            rewritten_chunks.append(rewritten[:140])
        return rewritten_chunks

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

    def rewrite_subtitle_chunks(
        self,
        job: JobState,
        source: SourceAsset,
        segment: SegmentCandidate,
        chunks: list[str],
    ) -> list[str]:
        llm = self._active_llm()
        if llm:
            return llm.rewrite_subtitle_chunks(job, source, segment, chunks) or chunks
        return chunks
