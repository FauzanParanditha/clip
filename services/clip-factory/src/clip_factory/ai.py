from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from urllib import error, request

from .contracts import JobState, SegmentCandidate, SourceAsset


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

