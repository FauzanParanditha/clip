from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .contracts import IntakeRequest
from .pipeline import ClipPipeline
from .storage import JsonJobStore


class IntakeJobModel(BaseModel):
    source_url: str
    content_type: str = Field(default="podcast")
    platform_targets: list[str] = Field(default_factory=lambda: ["youtube_shorts", "tiktok"])
    language_mode: str = "same_as_source"
    style: str = "clean_editorial"
    clip_count_target: int = Field(default=8, ge=5, le=12)
    auto_publish: bool = False


settings = Settings.from_env()
store = JsonJobStore(settings.data_dir)
pipeline = ClipPipeline(settings, store)
app = FastAPI(title="AI Clip Factory", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/jobs")
def create_job(payload: IntakeJobModel) -> dict:
    job = pipeline.create_job(
        IntakeRequest(
            source_url=payload.source_url,
            content_type=payload.content_type,  # type: ignore[arg-type]
            platform_targets=payload.platform_targets,  # type: ignore[arg-type]
            language_mode=payload.language_mode,
            style=payload.style,
            clip_count_target=payload.clip_count_target,
            auto_publish=payload.auto_publish,
        )
    )
    return job.to_dict()


@app.post("/v1/jobs/{job_id}/ingest")
def ingest_job(job_id: str) -> dict:
    return _run_step(lambda: pipeline.run_ingest_step(job_id).to_dict())


@app.post("/v1/jobs/{job_id}/transcript")
def transcribe_job(job_id: str) -> dict:
    return _run_step(lambda: pipeline.run_transcript_step(job_id).to_dict())


@app.post("/v1/jobs/{job_id}/rank")
def rank_job(job_id: str) -> dict:
    return _run_step(lambda: pipeline.run_rank_step(job_id).to_dict())


@app.post("/v1/jobs/{job_id}/select")
def select_job(job_id: str) -> dict:
    return _run_step(lambda: pipeline.run_select_step(job_id).to_dict())


@app.post("/v1/jobs/{job_id}/clips/{clip_id}/enqueue-render")
def enqueue_render(job_id: str, clip_id: str) -> dict:
    return _run_step(lambda: pipeline.enqueue_render(job_id, clip_id).to_dict())


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _run_step(lambda: pipeline.get_job(job_id).to_dict())


@app.get("/v1/jobs/{job_id}/manifest")
def get_manifest(job_id: str) -> dict:
    return _run_step(lambda: pipeline.get_manifest(job_id))


@app.get("/v1/jobs/{job_id}/clips/{clip_id}")
def get_clip(job_id: str, clip_id: str) -> dict:
    return _run_step(lambda: pipeline.get_clip(job_id, clip_id).to_dict())


def _run_step(callback):
    try:
        return callback()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

