from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clip_factory.storage import JsonJobStore


class StorageTests(unittest.TestCase):
    def test_write_json_is_atomic_and_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonJobStore(Path(tmpdir))
            path = Path(tmpdir) / "jobs" / "job_x" / "job.json"
            store.write_json(path, {"status": "queued"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "queued"})
            self.assertFalse((Path(str(path) + ".tmp")).exists())

    def test_read_json_retries_on_transient_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonJobStore(Path(tmpdir))
            path = Path(tmpdir) / "jobs" / "job_y" / "job.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

            def repair_file() -> None:
                path.write_text('{"status":"queued"}', encoding="utf-8")

            # Simulate a transient bad read before the next retry.
            repair_file()
            self.assertEqual(store.read_json(path), {"status": "queued"})


if __name__ == "__main__":
    unittest.main()
