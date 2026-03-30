from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clip_factory.ai import GeminiHybridScorer, OpenAIHybridScorer, _strict_json_schema
from clip_factory.config import Settings
from clip_factory.contracts import ClipMetadata, IntakeRequest, JobState, SegmentCandidate, SourceAsset


class TestableOpenAIHybridScorer(OpenAIHybridScorer):
    def __init__(self, settings: Settings, response_payload: dict | None) -> None:
        super().__init__(settings)
        self._response_payload = response_payload

    def _generate_json(self, **kwargs):  # type: ignore[override]
        return self._response_payload


class TestableGeminiHybridScorer(GeminiHybridScorer):
    def __init__(self, settings: Settings, response_payload: dict | None) -> None:
        super().__init__(settings)
        self._response_payload = response_payload

    def _generate_json(self, **kwargs):  # type: ignore[override]
        return self._response_payload


def build_settings() -> Settings:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        return Settings(
            data_dir=root / "data",
            output_dir=root / "outputs",
            redis_url="redis://localhost:6379/0",
            queue_key="clip-factory:render",
            ai_scorer_url=None,
            ai_scorer_bearer_token=None,
            llm_provider="auto",
            openai_api_key="test-openai-key",
            openai_model="gpt-5-mini",
            openai_base_url="https://api.openai.com/v1/responses",
            openai_timeout_seconds=60,
            openai_reasoning_effort=None,
            gemini_api_key="test-gemini-key",
            gemini_model="gemini-2.5-flash",
            gemini_base_url="https://generativelanguage.googleapis.com/v1beta",
            gemini_timeout_seconds=60,
            llm_subtitle_cleanup_enabled=True,
            llm_subtitle_cleanup_max_chunks=16,
            fallback_clip_count=8,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="int8",
            ffmpeg_binary="ffmpeg",
            ytdlp_binary="yt-dlp",
            subtitle_font="Arial",
        )


class SharedLLMHybridScorerAssertions:
    def assert_candidate_enrichment(self, scorer) -> None:
        job = JobState(job_id="job_1", input=IntakeRequest(source_url="https://example.com"))
        source = SourceAsset(source_url="https://example.com", title="Video sumber", language="id")
        candidates = [
            SegmentCandidate(
                segment_id="seg_1",
                start_ms=0,
                end_ms=30000,
                score=60.0,
                reason="heuristic",
                hook_text="Hook lama",
                keywords=["lama"],
                confidence=0.9,
                text="Ini segmen penting untuk testing hybrid scorer.",
            )
        ]

        enriched = scorer.enrich_candidates(job, source, candidates)
        self.assertEqual(len(enriched), 1)
        self.assertAlmostEqual(enriched[0].score, 77.6, places=1)
        self.assertEqual(enriched[0].hook_text, "Ini bagian paling kuat buat dijadiin shorts")
        self.assertEqual(enriched[0].keywords, ["workflow", "ai", "coding"])

    def assert_metadata_enrichment(self, scorer) -> None:
        job = JobState(job_id="job_1", input=IntakeRequest(source_url="https://example.com"))
        source = SourceAsset(source_url="https://example.com", title="Video sumber", language="id")
        segment = SegmentCandidate(
            segment_id="seg_1",
            start_ms=0,
            end_ms=30000,
            score=80.0,
            reason="heuristic",
            hook_text="Hook lama",
            keywords=["lama"],
            confidence=0.9,
            text="Ini segmen penting untuk testing metadata rewrite.",
        )
        fallback = ClipMetadata(
            hook_text="Fallback hook",
            titles=["Fallback 1", "Fallback 2", "Fallback 3"],
            caption="Fallback caption",
            hashtags=["#fallback"],
            highlight_keywords=["fallback"],
            source_timestamp_label="00:00",
            selection_reason="Fallback reason",
        )

        result = scorer.enrich_clip_metadata(job, source, {"clip_01": (segment, fallback), "clip_02": (segment, fallback)})
        self.assertEqual(result["clip_01"].hook_text, "Hook AI yang lebih nendang")
        self.assertEqual(result["clip_01"].titles[0], "Judul AI 1")
        self.assertEqual(result["clip_01"].caption, "Caption AI yang lebih natural")
        self.assertIn("#ai", result["clip_01"].hashtags)
        self.assertEqual(result["clip_02"].hook_text, "Fallback hook")
        self.assertEqual(result["clip_02"].titles[0], "Fallback 1")

    def assert_subtitle_cleanup(self, scorer) -> None:
        job = JobState(job_id="job_1", input=IntakeRequest(source_url="https://example.com"))
        source = SourceAsset(source_url="https://example.com", title="Video sumber", language="id")
        segment = SegmentCandidate(
            segment_id="seg_1",
            start_ms=0,
            end_ms=30000,
            score=80.0,
            reason="heuristic",
            hook_text="Hook lama",
            keywords=["lama"],
            confidence=0.9,
            text="Gua udah nyobain tool ini dan hasilnya jauh lebih stabil buat bikin aplikasi.",
        )
        chunks = [
            "gua udah nyobain tul ini",
            "dan hasilnya jauh lebih setabil",
        ]

        result = scorer.rewrite_subtitle_chunks(job, source, segment, chunks)
        self.assertEqual(
            result,
            [
                "Gue udah nyobain tool ini",
                "dan hasilnya jauh lebih stabil",
            ],
        )


class OpenAIHybridScorerTests(unittest.TestCase, SharedLLMHybridScorerAssertions):
    def test_strict_json_schema_adds_additional_properties_false_recursively(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "clips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "titles": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["titles"],
                    },
                }
            },
            "required": ["clips"],
        }

        strict_schema = _strict_json_schema(schema)
        self.assertFalse(strict_schema["additionalProperties"])
        self.assertFalse(strict_schema["properties"]["clips"]["items"]["additionalProperties"])

    def test_enrich_candidates_blends_llm_and_heuristic_scores(self) -> None:
        scorer = TestableOpenAIHybridScorer(
            build_settings(),
            {
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "score": 92,
                        "reason": "Strong payoff and clear standalone setup",
                        "hook_text": "Ini bagian paling kuat buat dijadiin shorts",
                        "keywords": ["workflow", "ai", "coding"],
                    }
                ]
            },
        )
        self.assert_candidate_enrichment(scorer)

    def test_enrich_clip_metadata_uses_llm_rewrite_and_fallbacks(self) -> None:
        scorer = TestableOpenAIHybridScorer(
            build_settings(),
            {
                "clips": [
                    {
                        "clip_id": "clip_01",
                        "hook_text": "Hook AI yang lebih nendang",
                        "titles": ["Judul AI 1", "Judul AI 2", "Judul AI 3"],
                        "caption": "Caption AI yang lebih natural",
                        "hashtags": ["#ai", "#workflow", "#shorts"],
                        "highlight_keywords": ["workflow", "automation", "shorts"],
                        "selection_reason": "Bagian ini paling jelas payoff-nya.",
                    }
                ]
            },
        )
        self.assert_metadata_enrichment(scorer)

    def test_rewrite_subtitle_chunks_cleans_asr_output(self) -> None:
        scorer = TestableOpenAIHybridScorer(
            build_settings(),
            {
                "subtitle_chunks": [
                    {"chunk_id": "chunk_01", "text": "Gue udah nyobain tool ini"},
                    {"chunk_id": "chunk_02", "text": "dan hasilnya jauh lebih stabil"},
                ]
            },
        )
        self.assert_subtitle_cleanup(scorer)


class GeminiHybridScorerTests(unittest.TestCase, SharedLLMHybridScorerAssertions):
    def test_enrich_candidates_blends_llm_and_heuristic_scores(self) -> None:
        settings = build_settings()
        settings.openai_api_key = None
        scorer = TestableGeminiHybridScorer(
            settings,
            {
                "segments": [
                    {
                        "segment_id": "seg_1",
                        "score": 92,
                        "reason": "Strong payoff and clear standalone setup",
                        "hook_text": "Ini bagian paling kuat buat dijadiin shorts",
                        "keywords": ["workflow", "ai", "coding"],
                    }
                ]
            },
        )
        self.assert_candidate_enrichment(scorer)

    def test_enrich_clip_metadata_uses_llm_rewrite_and_fallbacks(self) -> None:
        settings = build_settings()
        settings.openai_api_key = None
        scorer = TestableGeminiHybridScorer(
            settings,
            {
                "clips": [
                    {
                        "clip_id": "clip_01",
                        "hook_text": "Hook AI yang lebih nendang",
                        "titles": ["Judul AI 1", "Judul AI 2", "Judul AI 3"],
                        "caption": "Caption AI yang lebih natural",
                        "hashtags": ["#ai", "#workflow", "#shorts"],
                        "highlight_keywords": ["workflow", "automation", "shorts"],
                        "selection_reason": "Bagian ini paling jelas payoff-nya.",
                    }
                ]
            },
        )
        self.assert_metadata_enrichment(scorer)

    def test_rewrite_subtitle_chunks_cleans_asr_output(self) -> None:
        settings = build_settings()
        settings.openai_api_key = None
        scorer = TestableGeminiHybridScorer(
            settings,
            {
                "subtitle_chunks": [
                    {"chunk_id": "chunk_01", "text": "Gue udah nyobain tool ini"},
                    {"chunk_id": "chunk_02", "text": "dan hasilnya jauh lebih stabil"},
                ]
            },
        )
        self.assert_subtitle_cleanup(scorer)


if __name__ == "__main__":
    unittest.main()
