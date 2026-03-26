from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import Settings
from .contracts import SourceAsset
from .storage import JsonJobStore


def run_ingest(job_id: str, source_url: str, settings: Settings, store: JsonJobStore) -> SourceAsset:
    source_dir = store.source_dir(job_id)
    metadata_path = source_dir / "source.metadata.json"
    output_template = source_dir / "source.%(ext)s"

    metadata_command = [
        settings.ytdlp_binary,
        "--dump-single-json",
        "--no-playlist",
        source_url,
    ]
    metadata_result = subprocess.run(metadata_command, check=True, capture_output=True, text=True)
    metadata = json.loads(metadata_result.stdout)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

    download_command = [
        settings.ytdlp_binary,
        "--no-playlist",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_template),
        source_url,
    ]
    subprocess.run(download_command, check=True, capture_output=True, text=True)

    video_file = _find_source_video(source_dir)
    audio_path = source_dir / "source.wav"
    extract_audio_command = [
        settings.ffmpeg_binary,
        "-y",
        "-i",
        str(video_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    subprocess.run(extract_audio_command, check=True, capture_output=True, text=True)

    duration_seconds = float(metadata.get("duration") or 0)
    return SourceAsset(
        source_url=source_url,
        title=metadata.get("title") or "",
        description=metadata.get("description") or "",
        uploader=metadata.get("uploader") or metadata.get("channel") or "",
        duration_ms=int(duration_seconds * 1000),
        language=metadata.get("language"),
        video_path=str(video_file),
        audio_path=str(audio_path),
        metadata_path=str(metadata_path),
    )


def _find_source_video(source_dir: Path) -> Path:
    candidates = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() != ".json" and path.name != "source.wav"]
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce a video artifact")
    return sorted(candidates)[0]

