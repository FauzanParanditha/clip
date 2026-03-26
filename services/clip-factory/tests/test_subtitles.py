import unittest

from clip_factory.contracts import TranscriptWord
from clip_factory.subtitles import render_ass


class SubtitleTests(unittest.TestCase):
    def test_render_ass_contains_hook_and_highlight(self) -> None:
        words = [
            TranscriptWord("This", 0, 250),
            TranscriptWord("rate", 260, 500),
            TranscriptWord("shock", 510, 760),
            TranscriptWord("changes", 770, 1010),
            TranscriptWord("everything.", 1020, 1290),
        ]
        rendered = render_ass(words, hook_text="This rate shock changes everything", keywords=["rate", "shock"])
        self.assertIn("Style: Clip", rendered)
        self.assertIn("Dialogue: 0,0:00:00.00", rendered)
        self.assertIn(r"{\c&H2BFFFF&\b1}This", rendered)
        self.assertIn(r"{\c&H90F5FF&}rate", rendered)

    def test_render_ass_uses_multiline_layout(self) -> None:
        words = [
            TranscriptWord("Markets", 0, 200),
            TranscriptWord("usually", 210, 400),
            TranscriptWord("react", 410, 610),
            TranscriptWord("late", 620, 820),
            TranscriptWord("to", 830, 920),
            TranscriptWord("policy", 930, 1120),
            TranscriptWord("changes", 1130, 1330),
        ]
        rendered = render_ass(words, hook_text="", keywords=["policy"])
        self.assertIn(r"\N", rendered)


if __name__ == "__main__":
    unittest.main()

