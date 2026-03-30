from __future__ import annotations

import json
import os
import tempfile
import time
from json import JSONDecodeError
from pathlib import Path

from .contracts import ClipState, JobState, TranscriptDocument


class JsonJobStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.jobs_dir = self.root_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_file(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def source_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "source"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def transcript_file(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "transcript.json"

    def clips_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "clips"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clip_file(self, job_id: str, clip_id: str) -> Path:
        return self.clips_dir(job_id) / f"{clip_id}.json"

    def artifacts_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_json_file(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def manifest_csv_file(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.csv"

    def write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(data, indent=2, ensure_ascii=True))
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)

    def read_json(self, path: Path) -> dict:
        last_error: Exception | None = None
        for _ in range(5):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.02)
        if last_error:
            raise last_error
        raise FileNotFoundError(path)

    def save_job(self, job: JobState) -> None:
        self.write_json(self.job_file(job.job_id), job.to_dict())

    def load_job(self, job_id: str) -> JobState:
        return JobState.from_dict(self.read_json(self.job_file(job_id)))

    def save_transcript(self, job_id: str, transcript: TranscriptDocument) -> Path:
        path = self.transcript_file(job_id)
        self.write_json(path, transcript.to_dict())
        return path

    def load_transcript(self, job_id: str) -> TranscriptDocument:
        return TranscriptDocument.from_dict(self.read_json(self.transcript_file(job_id)))

    def save_clip(self, clip: ClipState) -> None:
        self.write_json(self.clip_file(clip.job_id, clip.clip_id), clip.to_dict())

    def load_clip(self, job_id: str, clip_id: str) -> ClipState:
        return ClipState.from_dict(self.read_json(self.clip_file(job_id, clip_id)))

    def list_clips(self, job_id: str) -> list[ClipState]:
        clips = []
        for path in sorted(self.clips_dir(job_id).glob("*.json")):
            clips.append(ClipState.from_dict(self.read_json(path)))
        return clips
