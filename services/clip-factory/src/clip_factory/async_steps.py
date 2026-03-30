from __future__ import annotations

import threading
from collections.abc import Callable


class BackgroundStepRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def start(self, key: str, callback: Callable[[], object]) -> bool:
        with self._lock:
            existing = self._threads.get(key)
            if existing and existing.is_alive():
                return False

            thread = threading.Thread(
                target=self._run,
                args=(key, callback),
                daemon=True,
                name=f"clip-step-{key}",
            )
            self._threads[key] = thread
            thread.start()
            return True

    def is_running(self, key: str) -> bool:
        with self._lock:
            thread = self._threads.get(key)
            return bool(thread and thread.is_alive())

    def _run(self, key: str, callback: Callable[[], object]) -> None:
        try:
            callback()
        finally:
            with self._lock:
                thread = self._threads.get(key)
                if thread is threading.current_thread():
                    self._threads.pop(key, None)
