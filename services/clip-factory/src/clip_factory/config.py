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
    llm_provider: str
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str
    openai_timeout_seconds: int
    openai_reasoning_effort: str | None
    gemini_api_key: str | None
    gemini_model: str
    gemini_base_url: str
    gemini_timeout_seconds: int
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
            llm_provider=os.getenv("CLIP_FACTORY_LLM_PROVIDER", "auto").lower(),
            openai_api_key=os.getenv("CLIP_FACTORY_OPENAI_API_KEY") or None,
            openai_model=os.getenv("CLIP_FACTORY_OPENAI_MODEL", "gpt-5-mini"),
            openai_base_url=os.getenv("CLIP_FACTORY_OPENAI_BASE_URL", "https://api.openai.com/v1/responses"),
            openai_timeout_seconds=max(5, int(os.getenv("CLIP_FACTORY_OPENAI_TIMEOUT_SECONDS", "60"))),
            openai_reasoning_effort=os.getenv("CLIP_FACTORY_OPENAI_REASONING_EFFORT") or None,
            gemini_api_key=os.getenv("CLIP_FACTORY_GEMINI_API_KEY") or None,
            gemini_model=os.getenv("CLIP_FACTORY_GEMINI_MODEL", "gemini-2.5-flash"),
            gemini_base_url=os.getenv("CLIP_FACTORY_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            gemini_timeout_seconds=max(5, int(os.getenv("CLIP_FACTORY_GEMINI_TIMEOUT_SECONDS", "60"))),
            fallback_clip_count=max(5, min(12, int(os.getenv("CLIP_FACTORY_FALLBACK_CLIP_COUNT", "8")))),
            whisper_model=os.getenv("CLIP_FACTORY_WHISPER_MODEL", "small"),
            whisper_device=os.getenv("CLIP_FACTORY_WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("CLIP_FACTORY_WHISPER_COMPUTE_TYPE", "int8"),
            ffmpeg_binary=os.getenv("CLIP_FACTORY_FFMPEG_BINARY", "ffmpeg"),
            ytdlp_binary=os.getenv("CLIP_FACTORY_YTDLP_BINARY", "yt-dlp"),
            subtitle_font=os.getenv("CLIP_FACTORY_SUBTITLE_FONT", "Arial"),
        )
