import unittest

from clip_factory.contracts import SegmentCandidate, TranscriptDocument, TranscriptWord
from clip_factory.heuristics import build_candidate_segments, score_segment, select_segments


def make_words(text: str, start_ms: int = 0, step_ms: int = 550, confidence: float = 0.92) -> list[TranscriptWord]:
    words = []
    current = start_ms
    for raw in text.split():
        words.append(
            TranscriptWord(
                text=raw,
                start_ms=current,
                end_ms=current + step_ms - 120,
                confidence=confidence,
            )
        )
        current += step_ms
    return words


class HeuristicsTests(unittest.TestCase):
    def test_score_segment_penalizes_sponsor_language(self) -> None:
        clean_score, _, _ = score_segment(
            "This is the critical point why rates matter right now.",
            duration_ms=32_000,
            confidence=0.92,
            start_ms=120_000,
            total_duration_ms=3_600_000,
        )
        sponsor_score, _, flags = score_segment(
            "Before we start this sponsor code gives you a discount and promo.",
            duration_ms=32_000,
            confidence=0.92,
            start_ms=120_000,
            total_duration_ms=3_600_000,
        )
        self.assertGreater(clean_score, sponsor_score)
        self.assertIn("sponsor_risk", flags)

    def test_build_candidate_segments_returns_ranked_candidates(self) -> None:
        transcript = TranscriptDocument(
            language="en",
            average_confidence=0.91,
            words=make_words(
                (
                    "The biggest mistake founders make is scaling too early. "
                    "If revenue is not repeatable you just scale the chaos. "
                    "That is why the timing of hiring matters more than people think. "
                    "When you fix retention first the growth channel gets dramatically cheaper. "
                    "That one change can give you a much stronger payback period."
                ),
                step_ms=700,
            ),
        )
        candidates = build_candidate_segments(transcript, target_count=5)
        self.assertTrue(candidates)
        self.assertGreater(candidates[0].score, 0)
        self.assertLess(candidates[0].start_ms, candidates[0].end_ms)

    def test_news_score_prefers_title_aligned_clip_over_side_angle_reaction(self) -> None:
        title = "Trump accused of secret plan for Iran ground invasion as thousands of US Marines arrive"
        headline_score, _, _ = score_segment(
            (
                "Thousands of US Marines have arrived as reports say Trump could approve "
                "ground operations in Iran within days."
            ),
            duration_ms=36_000,
            confidence=0.92,
            start_ms=90_000,
            total_duration_ms=900_000,
            content_type="news",
            source_title=title,
        )
        side_angle_score, _, flags = score_segment(
            "A local congregation says Iran is a threat and public opinion may change over time.",
            duration_ms=36_000,
            confidence=0.92,
            start_ms=760_000,
            total_duration_ms=900_000,
            content_type="news",
            source_title=title,
        )
        self.assertGreater(headline_score, side_angle_score)
        self.assertIn("side_angle", flags)

    def test_news_selection_avoids_duplicate_headline_facets(self) -> None:
        candidates = [
            SegmentCandidate(
                "a",
                0,
                30_000,
                88.0,
                "headline aligned",
                "Clip A",
                ["darurat", "energi", "apbn", "kuat"],
                0.9,
                "A",
            ),
            SegmentCandidate(
                "b",
                40_000,
                70_000,
                86.0,
                "headline aligned",
                "Clip B",
                ["energi", "apbn", "terkendali", "risiko"],
                0.9,
                "B",
            ),
            SegmentCandidate(
                "c",
                80_000,
                110_000,
                83.0,
                "headline aligned",
                "Clip C",
                ["pajak", "solusi", "ekonomi", "komprehensif"],
                0.9,
                "C",
            ),
            SegmentCandidate(
                "d",
                120_000,
                150_000,
                81.0,
                "headline aligned",
                "Clip D",
                ["kritik", "kredibilitas", "pengkritik", "kebijakan"],
                0.9,
                "D",
            ),
            SegmentCandidate(
                "e",
                160_000,
                190_000,
                80.0,
                "headline aligned",
                "Clip E",
                ["subsidi", "pasokan", "cadangan", "energi"],
                0.9,
                "E",
            ),
        ]
        selected, _ = select_segments(
            candidates,
            requested_count=5,
            transcript_confidence=0.9,
            content_type="news",
            source_title="Menkeu Purbaya Pastikan RI Belum Darurat Energi: APBN Kita Kuat",
        )
        selected_ids = {segment.segment_id for segment in selected}
        self.assertIn("a", selected_ids)
        self.assertNotIn("b", selected_ids)

    def test_select_segments_avoids_overlap(self) -> None:
        candidates = [
            SegmentCandidate("a", 0, 30_000, 84.0, "strong opener", "Clip A", ["economy", "rates"], 0.9, "A"),
            SegmentCandidate("b", 20_000, 48_000, 82.0, "strong opener", "Clip B", ["economy", "rates"], 0.9, "B"),
            SegmentCandidate("c", 60_000, 95_000, 81.0, "standalone", "Clip C", ["policy", "market"], 0.9, "C"),
            SegmentCandidate("d", 105_000, 140_000, 80.0, "standalone", "Clip D", ["inflation", "jobs"], 0.9, "D"),
            SegmentCandidate("e", 150_000, 185_000, 79.0, "standalone", "Clip E", ["crypto", "etf"], 0.9, "E"),
        ]
        selected, review_needed = select_segments(candidates, requested_count=5, transcript_confidence=0.9)
        self.assertFalse(review_needed)
        self.assertGreaterEqual(len(selected), 4)
        self.assertEqual(selected[0].segment_id, "a")
        self.assertNotIn("b", {segment.segment_id for segment in selected})


if __name__ == "__main__":
    unittest.main()
