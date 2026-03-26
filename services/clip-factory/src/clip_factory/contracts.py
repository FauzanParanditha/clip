from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


ContentType = Literal["podcast", "news", "other"]
PlatformTarget = Literal["youtube_shorts", "tiktok"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class IntakeRequest:
    source_url: str
    content_type: ContentType = "podcast"
    platform_targets: list[PlatformTarget] = field(default_factory=lambda: ["youtube_shorts", "tiktok"])
    language_mode: str = "same_as_source"
    style: str = "clean_editorial"
    clip_count_target: int = 8
    auto_publish: bool = False

    def normalized_clip_count(self) -> int:
        return max(5, min(12, self.clip_count_target))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "content_type": self.content_type,
            "platform_targets": self.platform_targets,
            "language_mode": self.language_mode,
            "style": self.style,
            "clip_count_target": self.normalized_clip_count(),
            "auto_publish": self.auto_publish,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntakeRequest":
        return cls(
            source_url=data["source_url"],
            content_type=data.get("content_type", "podcast"),
            platform_targets=list(data.get("platform_targets", ["youtube_shorts", "tiktok"])),
            language_mode=data.get("language_mode", "same_as_source"),
            style=data.get("style", "clean_editorial"),
            clip_count_target=int(data.get("clip_count_target", 8)),
            auto_publish=bool(data.get("auto_publish", False)),
        )


@dataclass(slots=True)
class SourceAsset:
    source_url: str
    title: str = ""
    description: str = ""
    uploader: str = ""
    duration_ms: int = 0
    language: str | None = None
    video_path: str | None = None
    audio_path: str | None = None
    metadata_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "title": self.title,
            "description": self.description,
            "uploader": self.uploader,
            "duration_ms": self.duration_ms,
            "language": self.language,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "metadata_path": self.metadata_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceAsset":
        return cls(
            source_url=data["source_url"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            uploader=data.get("uploader", ""),
            duration_ms=int(data.get("duration_ms", 0)),
            language=data.get("language"),
            video_path=data.get("video_path"),
            audio_path=data.get("audio_path"),
            metadata_path=data.get("metadata_path"),
        )


@dataclass(slots=True)
class TranscriptWord:
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptWord":
        return cls(
            text=data["text"],
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            confidence=float(data.get("confidence", 1.0)),
            speaker=data.get("speaker"),
        )


@dataclass(slots=True)
class TranscriptDocument:
    language: str | None
    average_confidence: float
    words: list[TranscriptWord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "average_confidence": self.average_confidence,
            "words": [word.to_dict() for word in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptDocument":
        return cls(
            language=data.get("language"),
            average_confidence=float(data.get("average_confidence", 0.0)),
            words=[TranscriptWord.from_dict(word) for word in data.get("words", [])],
        )


@dataclass(slots=True)
class SegmentCandidate:
    segment_id: str
    start_ms: int
    end_ms: int
    score: float
    reason: str
    hook_text: str
    keywords: list[str]
    confidence: float
    text: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "score": self.score,
            "reason": self.reason,
            "hook_text": self.hook_text,
            "keywords": self.keywords,
            "confidence": self.confidence,
            "text": self.text,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentCandidate":
        return cls(
            segment_id=data["segment_id"],
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            score=float(data.get("score", 0.0)),
            reason=data.get("reason", ""),
            hook_text=data.get("hook_text", ""),
            keywords=list(data.get("keywords", [])),
            confidence=float(data.get("confidence", 0.0)),
            text=data.get("text", ""),
            flags=list(data.get("flags", [])),
        )


@dataclass(slots=True)
class ClipMetadata:
    titles: list[str]
    caption: str
    hashtags: list[str]
    highlight_keywords: list[str]
    source_timestamp_label: str
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "titles": self.titles,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "highlight_keywords": self.highlight_keywords,
            "source_timestamp_label": self.source_timestamp_label,
            "selection_reason": self.selection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipMetadata":
        return cls(
            titles=list(data.get("titles", [])),
            caption=data.get("caption", ""),
            hashtags=list(data.get("hashtags", [])),
            highlight_keywords=list(data.get("highlight_keywords", [])),
            source_timestamp_label=data.get("source_timestamp_label", ""),
            selection_reason=data.get("selection_reason", ""),
        )


@dataclass(slots=True)
class TrackingBox:
    frame_ms: int
    frame_width: int
    frame_height: int
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_ms": self.frame_ms,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackingBox":
        return cls(
            frame_ms=int(data["frame_ms"]),
            frame_width=int(data["frame_width"]),
            frame_height=int(data["frame_height"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


@dataclass(slots=True)
class RenderRequest:
    job_id: str
    clip_id: str
    input_video_path: str
    output_path: str
    subtitle_path: str
    start_ms: int
    end_ms: int
    hook_text: str
    keywords: list[str]
    crop_mode: str = "auto_reframe"
    subtitle_mode: str = "word_timed"
    effects: list[str] = field(default_factory=lambda: ["active_word_highlight", "subtle_punch_in", "hook_card"])
    output_profile: str = "1080x1920_h264_aac"
    tracking_boxes: list[TrackingBox] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "clip_id": self.clip_id,
            "input_video_path": self.input_video_path,
            "output_path": self.output_path,
            "subtitle_path": self.subtitle_path,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "hook_text": self.hook_text,
            "keywords": self.keywords,
            "crop_mode": self.crop_mode,
            "subtitle_mode": self.subtitle_mode,
            "effects": self.effects,
            "output_profile": self.output_profile,
            "tracking_boxes": [box.to_dict() for box in self.tracking_boxes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderRequest":
        return cls(
            job_id=data["job_id"],
            clip_id=data["clip_id"],
            input_video_path=data["input_video_path"],
            output_path=data["output_path"],
            subtitle_path=data["subtitle_path"],
            start_ms=int(data["start_ms"]),
            end_ms=int(data["end_ms"]),
            hook_text=data.get("hook_text", ""),
            keywords=list(data.get("keywords", [])),
            crop_mode=data.get("crop_mode", "auto_reframe"),
            subtitle_mode=data.get("subtitle_mode", "word_timed"),
            effects=list(data.get("effects", ["active_word_highlight", "subtle_punch_in", "hook_card"])),
            output_profile=data.get("output_profile", "1080x1920_h264_aac"),
            tracking_boxes=[TrackingBox.from_dict(box) for box in data.get("tracking_boxes", [])],
        )


@dataclass(slots=True)
class ClipState:
    clip_id: str
    job_id: str
    segment: SegmentCandidate
    metadata: ClipMetadata | None = None
    status: str = "pending"
    output_path: str | None = None
    subtitle_path: str | None = None
    render_request: RenderRequest | None = None
    render_command: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "job_id": self.job_id,
            "segment": self.segment.to_dict(),
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "status": self.status,
            "output_path": self.output_path,
            "subtitle_path": self.subtitle_path,
            "render_request": self.render_request.to_dict() if self.render_request else None,
            "render_command": self.render_command,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipState":
        return cls(
            clip_id=data["clip_id"],
            job_id=data["job_id"],
            segment=SegmentCandidate.from_dict(data["segment"]),
            metadata=ClipMetadata.from_dict(data["metadata"]) if data.get("metadata") else None,
            status=data.get("status", "pending"),
            output_path=data.get("output_path"),
            subtitle_path=data.get("subtitle_path"),
            render_request=RenderRequest.from_dict(data["render_request"]) if data.get("render_request") else None,
            render_command=list(data.get("render_command", [])),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            error=data.get("error"),
        )


@dataclass(slots=True)
class JobState:
    job_id: str
    input: IntakeRequest
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    source: SourceAsset | None = None
    transcript_path: str | None = None
    transcript_language: str | None = None
    transcript_confidence: float | None = None
    candidate_segments: list[SegmentCandidate] = field(default_factory=list)
    selected_clip_ids: list[str] = field(default_factory=list)
    manifest_json_path: str | None = None
    manifest_csv_path: str | None = None
    review_needed: bool = False
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "input": self.input.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source.to_dict() if self.source else None,
            "transcript_path": self.transcript_path,
            "transcript_language": self.transcript_language,
            "transcript_confidence": self.transcript_confidence,
            "candidate_segments": [segment.to_dict() for segment in self.candidate_segments],
            "selected_clip_ids": self.selected_clip_ids,
            "manifest_json_path": self.manifest_json_path,
            "manifest_csv_path": self.manifest_csv_path,
            "review_needed": self.review_needed,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        return cls(
            job_id=data["job_id"],
            input=IntakeRequest.from_dict(data["input"]),
            status=data.get("status", "queued"),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            source=SourceAsset.from_dict(data["source"]) if data.get("source") else None,
            transcript_path=data.get("transcript_path"),
            transcript_language=data.get("transcript_language"),
            transcript_confidence=float(data["transcript_confidence"]) if data.get("transcript_confidence") is not None else None,
            candidate_segments=[SegmentCandidate.from_dict(segment) for segment in data.get("candidate_segments", [])],
            selected_clip_ids=list(data.get("selected_clip_ids", [])),
            manifest_json_path=data.get("manifest_json_path"),
            manifest_csv_path=data.get("manifest_csv_path"),
            review_needed=bool(data.get("review_needed", False)),
            last_error=data.get("last_error"),
        )

