from __future__ import annotations

import json
import sys
import time

from .config import Settings
from .pipeline import ClipPipeline
from .storage import JsonJobStore


def main() -> int:
    settings = Settings.from_env()
    store = JsonJobStore(settings.data_dir)
    pipeline = ClipPipeline(settings, store)
    client = pipeline.redis_client

    while True:
        item = client.blpop(settings.queue_key, timeout=5)
        if not item:
            time.sleep(0.5)
            continue
        _, raw_payload = item
        payload = json.loads(raw_payload)
        try:
            pipeline.process_render_message(payload)
        except Exception as exc:
            print(f"[clip-worker] render task failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
