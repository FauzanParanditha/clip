from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import cast

import redis

from .ai import HybridAISegmentScorer
from .config import Settings
from .contracts import ClipMetadata, ClipState, IntakeRequest, JobState, RenderRequest, SegmentCandidate, utc_now
from .heuristics import build_candidate_segments, generate_clip_metadata, select_segments, slice_words
from .ingest import run_ingest
from .rendering import render_clip
from .storage import JsonJobStore
from .subtitles import render_ass
from .tracking import detect_subject_tracking
from .transcribe import run_transcription


class ClipPipeline:
    def __init__(self, settings: Settings, store: JsonJobStore) -> None:
        self.settings = settings
        self.store = store
        self.redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.ai_scorer = HybridAISegmentScorer(settings)

    def create_job(self, intake: IntakeRequest) -> JobState:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        job = JobState(job_id=job_id, input=intake, status="queued")
        self.store.save_job(job)
        return job

    def get_job(self, job_id: str) -> JobState:
        return self.store.load_job(job_id)

    def get_clip(self, job_id: str, clip_id: str) -> ClipState:
        return self.store.load_clip(job_id, clip_id)

    def run_ingest_step(self, job_id: str) -> JobState:
        job = self.store.load_job(job_id)
        self._update_job(job, status="ingesting", error=None)
        try:
            source = run_ingest(job_id, job.input.source_url, self.settings, self.store)
            job.source = source
            self._update_job(job, status="ingested", error=None)
        except Exception as exc:
            self._update_job(job, status="failed", error=f"ingest failed: {exc}")
            raise
        return job

    def run_transcript_step(self, job_id: str) -> JobState:
        job = self.store.load_job(job_id)
        if not job.source or not job.source.audio_path:
            raise RuntimeError("Job has no ingested audio artifact")
        self._update_job(job, status="transcribing", error=None)
        try:
            transcript = run_transcription(job.source.audio_path, self.settings)
            transcript_path = self.store.save_transcript(job_id, transcript)
            job.transcript_path = str(transcript_path)
            job.transcript_language = transcript.language
            job.transcript_confidence = transcript.average_confidence
            self._update_job(job, status="transcribed", error=None)
        except Exception as exc:
            self._update_job(job, status="failed", error=f"transcription failed: {exc}")
            raise
        return job

    def run_rank_step(self, job_id: str) -> JobState:
        job = self.store.load_job(job_id)
        transcript = self.store.load_transcript(job_id)
        if not job.source:
            raise RuntimeError("Job source metadata missing")
        self._update_job(job, status="ranking", error=None)
        try:
            candidates = build_candidate_segments(transcript, job.input.normalized_clip_count())
            candidates = self.ai_scorer.enrich_candidates(job, job.source, candidates)
            job.candidate_segments = candidates[: max(24, job.input.normalized_clip_count() * 4)]
            self._update_job(job, status="ranked", error=None)
        except Exception as exc:
            self._update_job(job, status="failed", error=f"ranking failed: {exc}")
            raise
        return job

    def run_select_step(self, job_id: str) -> JobState:
        job = self.store.load_job(job_id)
        if not job.candidate_segments:
            raise RuntimeError("Job has no ranked candidates")
        if not job.source:
            raise RuntimeError("Job source metadata missing")
        transcript_confidence = job.transcript_confidence or 0.0
        selected, review_needed = select_segments(
            job.candidate_segments,
            requested_count=job.input.normalized_clip_count(),
            transcript_confidence=transcript_confidence,
        )
        if not selected:
            self._update_job(job, status="failed", error="No viable clip segments were selected")
            raise RuntimeError("No viable clip segments were selected")

        job.review_needed = review_needed
        job.selected_clip_ids = []
        fallback_by_clip_id: dict[str, tuple[SegmentCandidate, ClipMetadata]] = {}
        clip_rows: list[tuple[str, SegmentCandidate]] = []
        for index, segment in enumerate(selected, start=1):
            clip_id = f"clip_{index:02d}_{segment.segment_id[-6:]}"
            fallback_by_clip_id[clip_id] = (segment, generate_clip_metadata(segment, job.source))
            clip_rows.append((clip_id, segment))
            job.selected_clip_ids.append(clip_id)

        metadata_by_clip_id = self.ai_scorer.enrich_clip_metadata(job, job.source, fallback_by_clip_id)
        for clip_id, segment in clip_rows:
            clip = ClipState(
                clip_id=clip_id,
                job_id=job.job_id,
                segment=segment,
                metadata=cast(ClipMetadata, metadata_by_clip_id.get(clip_id, fallback_by_clip_id[clip_id][1])),
                status="pending",
            )
            self.store.save_clip(clip)

        self._update_job(job, status="render_pending", error=None)
        self.build_manifest(job_id)
        return job

    def enqueue_render(self, job_id: str, clip_id: str) -> ClipState:
        job = self.store.load_job(job_id)
        clip = self.store.load_clip(job_id, clip_id)
        clip.status = "queued"
        clip.updated_at = utc_now()
        clip.error = None
        self.store.save_clip(clip)
        self.redis_client.rpush(self.settings.queue_key, json.dumps({"job_id": job_id, "clip_id": clip_id}))
        self._update_job(job, status="rendering", error=None)
        self.build_manifest(job_id)
        return clip

    def process_render_message(self, payload: dict[str, str]) -> ClipState:
        job_id = payload["job_id"]
        clip_id = payload["clip_id"]
        job = self.store.load_job(job_id)
        clip = self.store.load_clip(job_id, clip_id)
        transcript = self.store.load_transcript(job_id)

        if not job.source or not job.source.video_path:
            raise RuntimeError("Job source video missing")

        clip.status = "rendering"
        clip.updated_at = utc_now()
        self.store.save_clip(clip)

        try:
            clip_words = slice_words(transcript.words, clip.segment.start_ms, clip.segment.end_ms, normalize=True)
            if not clip_words:
                raise RuntimeError("Selected segment has no transcript words")

            artifact_dir = self.store.artifacts_dir(job_id)
            subtitle_path = artifact_dir / f"{clip_id}.ass"
            subtitle_text = render_ass(
                clip_words,
                hook_text=clip.segment.hook_text,
                keywords=(clip.metadata.highlight_keywords if clip.metadata else clip.segment.keywords),
                font_name=self.settings.subtitle_font,
            )
            subtitle_path.write_text(subtitle_text, encoding="utf-8")

            tracking_boxes = detect_subject_tracking(
                job.source.video_path,
                clip.segment.start_ms,
                clip.segment.end_ms,
                metadata_path=job.source.metadata_path,
            )
            output_dir = self.settings.output_dir / job_id
            output_path = output_dir / f"{clip_id}.mp4"
            render_request = RenderRequest(
                job_id=job_id,
                clip_id=clip_id,
                input_video_path=job.source.video_path,
                output_path=str(output_path),
                subtitle_path=str(subtitle_path),
                start_ms=clip.segment.start_ms,
                end_ms=clip.segment.end_ms,
                hook_text=clip.segment.hook_text,
                keywords=(clip.metadata.highlight_keywords if clip.metadata else clip.segment.keywords),
                crop_mode="auto_reframe" if tracking_boxes else "contain",
                tracking_boxes=tracking_boxes,
            )
            command = render_clip(self.settings, render_request)

            clip.render_request = render_request
            clip.render_command = command
            clip.subtitle_path = str(subtitle_path)
            clip.output_path = str(output_path)
            clip.status = "review_needed" if job.review_needed else "ready"
            clip.updated_at = utc_now()
            clip.error = None
            self.store.save_clip(clip)
        except Exception as exc:
            clip.status = "failed"
            clip.error = f"render failed: {exc}"
            clip.updated_at = utc_now()
            self.store.save_clip(clip)
            self._refresh_job_status(job_id)
            self.build_manifest(job_id)
            raise

        self._refresh_job_status(job_id)
        self.build_manifest(job_id)
        return clip

    def build_manifest(self, job_id: str) -> dict:
        job = self.store.load_job(job_id)
        clips = self.store.list_clips(job_id)
        manifest = {
            "job_id": job.job_id,
            "status": job.status,
            "review_needed": job.review_needed,
            "source": job.source.to_dict() if job.source else None,
            "clips": [
                {
                    "clip_id": clip.clip_id,
                    "status": clip.status,
                    "start_ms": clip.segment.start_ms,
                    "end_ms": clip.segment.end_ms,
                    "score": clip.segment.score,
                    "reason": clip.segment.reason,
                    "hook_text": clip.segment.hook_text,
                    "keywords": clip.segment.keywords,
                    "output_path": clip.output_path,
                    "subtitle_path": clip.subtitle_path,
                    "titles": clip.metadata.titles if clip.metadata else [],
                    "caption": clip.metadata.caption if clip.metadata else "",
                    "hashtags": clip.metadata.hashtags if clip.metadata else [],
                    "source_timestamp_label": clip.metadata.source_timestamp_label if clip.metadata else "",
                }
                for clip in clips
            ],
        }

        manifest_json = self.store.manifest_json_file(job_id)
        manifest_csv = self.store.manifest_csv_file(job_id)
        self.store.write_json(manifest_json, manifest)
        self._write_manifest_csv(manifest_csv, clips)

        job.manifest_json_path = str(manifest_json)
        job.manifest_csv_path = str(manifest_csv)
        self._update_job(job, status=job.status, error=job.last_error)
        return manifest

    def get_manifest(self, job_id: str) -> dict:
        manifest_path = self.store.manifest_json_file(job_id)
        if not manifest_path.exists():
            return self.build_manifest(job_id)
        return self.store.read_json(manifest_path)

    def _write_manifest_csv(self, path: Path, clips: list[ClipState]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "clip_id",
                    "status",
                    "start_ms",
                    "end_ms",
                    "score",
                    "output_path",
                    "title_1",
                    "title_2",
                    "title_3",
                    "caption",
                    "hashtags",
                    "reason",
                ],
            )
            writer.writeheader()
            for clip in clips:
                titles = clip.metadata.titles if clip.metadata else []
                writer.writerow(
                    {
                        "clip_id": clip.clip_id,
                        "status": clip.status,
                        "start_ms": clip.segment.start_ms,
                        "end_ms": clip.segment.end_ms,
                        "score": clip.segment.score,
                        "output_path": clip.output_path or "",
                        "title_1": titles[0] if len(titles) > 0 else "",
                        "title_2": titles[1] if len(titles) > 1 else "",
                        "title_3": titles[2] if len(titles) > 2 else "",
                        "caption": clip.metadata.caption if clip.metadata else "",
                        "hashtags": " ".join(clip.metadata.hashtags) if clip.metadata else "",
                        "reason": clip.metadata.selection_reason if clip.metadata else clip.segment.reason,
                    }
                )

    def _refresh_job_status(self, job_id: str) -> JobState:
        job = self.store.load_job(job_id)
        clips = self.store.list_clips(job_id)
        statuses = {clip.status for clip in clips}
        if not clips:
            return job
        if statuses.issubset({"ready"}):
            job.status = "ready"
        elif statuses.issubset({"ready", "review_needed"}):
            job.status = "review_needed"
        elif "rendering" in statuses or "queued" in statuses or "pending" in statuses:
            job.status = "rendering"
        elif statuses == {"failed"}:
            job.status = "failed"
        elif "failed" in statuses:
            job.status = "partial"
        job.updated_at = utc_now()
        self.store.save_job(job)
        return job

    def _update_job(self, job: JobState, status: str, error: str | None) -> None:
        job.status = status
        job.last_error = error
        job.updated_at = utc_now()
        self.store.save_job(job)
