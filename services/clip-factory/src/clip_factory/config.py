from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    data_dir: Path
    output_dir: Path
    redis_url: str
    queue_key: str
    ai_scorer_url: str | None
    ai_scorer_bearer_token: str | None
    fallback_clip_count: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    ffmpeg_binary: str
    ytdlp_binary: str
    subtitle_font: str

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("CLIP_FACTORY_DATA_DIR", "./data")).resolve()
        output_dir = Path(os.getenv("CLIP_FACTORY_OUTPUT_DIR", "./outputs")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            data_dir=data_dir,
            output_dir=output_dir,
            redis_url=os.getenv("CLIP_FACTORY_REDIS_URL", "redis://localhost:6379/0"),
            queue_key=os.getenv("CLIP_FACTORY_QUEUE_KEY", "clip-factory:render"),
            ai_scorer_url=os.getenv("CLIP_FACTORY_AI_SCORER_URL") or None,
            ai_scorer_bearer_token=os.getenv("CLIP_FACTORY_AI_SCORER_BEARER_TOKEN") or None,
            fallback_clip_count=max(5, min(12, int(os.getenv("CLIP_FACTORY_FALLBACK_CLIP_COUNT", "8")))),
            whisper_model=os.getenv("CLIP_FACTORY_WHISPER_MODEL", "small"),
            whisper_device=os.getenv("CLIP_FACTORY_WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("CLIP_FACTORY_WHISPER_COMPUTE_TYPE", "int8"),
            ffmpeg_binary=os.getenv("CLIP_FACTORY_FFMPEG_BINARY", "ffmpeg"),
            ytdlp_binary=os.getenv("CLIP_FACTORY_YTDLP_BINARY", "yt-dlp"),
            subtitle_font=os.getenv("CLIP_FACTORY_SUBTITLE_FONT", "Arial"),
        )

