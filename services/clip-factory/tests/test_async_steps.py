from __future__ import annotations

import threading
import time
import unittest

from clip_factory.async_steps import BackgroundStepRunner


class BackgroundStepRunnerTest(unittest.TestCase):
    def test_prevents_duplicate_runs_while_step_is_active(self) -> None:
        runner = BackgroundStepRunner()
        started = threading.Event()
        release = threading.Event()

        def callback() -> None:
            started.set()
            release.wait(timeout=2)

        self.assertTrue(runner.start("job-1:transcript", callback))
        self.assertTrue(started.wait(timeout=1))
        self.assertTrue(runner.is_running("job-1:transcript"))
        self.assertFalse(runner.start("job-1:transcript", callback))

        release.set()
        time.sleep(0.05)
        self.assertFalse(runner.is_running("job-1:transcript"))

    def test_allows_restart_after_previous_run_finishes(self) -> None:
        runner = BackgroundStepRunner()
        calls: list[str] = []

        def callback() -> None:
            calls.append("ran")

        self.assertTrue(runner.start("job-2:ingest", callback))
        time.sleep(0.05)
        self.assertTrue(runner.start("job-2:ingest", callback))
        time.sleep(0.05)
        self.assertEqual(calls, ["ran", "ran"])


if __name__ == "__main__":
    unittest.main()
