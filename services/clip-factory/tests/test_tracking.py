from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clip_factory.tracking import source_uses_av1


class TrackingTests(unittest.TestCase):
    def test_source_uses_av1_from_requested_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.metadata.json"
            path.write_text(
                json.dumps(
                    {
                        "requested_formats": [
                            {"vcodec": "av01.0.08M.08"},
                            {"acodec": "mp4a.40.2"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(source_uses_av1(str(path)))

    def test_source_uses_av1_false_for_h264(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.metadata.json"
            path.write_text(json.dumps({"vcodec": "avc1.640028"}), encoding="utf-8")
            self.assertFalse(source_uses_av1(str(path)))


if __name__ == "__main__":
    unittest.main()
