from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Settings
from .contracts import RenderRequest, TrackingBox


def estimate_focus_ratio(tracking_boxes: list[TrackingBox]) -> float | None:
    if not tracking_boxes:
        return None
    weighted_center = 0.0
    weight_total = 0.0
    for box in tracking_boxes:
        if box.frame_width <= 0:
            continue
        center_ratio = (box.x + box.width / 2) / box.frame_width
        weight = max(1.0, box.width * box.height)
        weighted_center += center_ratio * weight
        weight_total += weight
    if not weight_total:
        return None
    return max(0.0, min(1.0, weighted_center / weight_total))


def escape_filter_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def build_vertical_crop_filter(focus_ratio: float | None = None) -> str:
    crop_w = "if(gte(iw/ih,9/16),ih*9/16,iw)"
    crop_h = "if(gte(iw/ih,9/16),ih,iw*16/9)"
    if focus_ratio is None:
        crop_x = "(iw-ow)/2"
    else:
        crop_x = f"max(0,min(iw-ow,iw*{focus_ratio:.4f}-ow/2))"
    crop_y = "(ih-oh)/2"
    return f"crop='{crop_w}':'{crop_h}':'{crop_x}':'{crop_y}'"


def build_contain_pad_filter() -> str:
    return ",".join(
        [
            "scale=1080:1920:force_original_aspect_ratio=decrease",
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=0x101010",
        ]
    )


def build_filter_chain(render_request: RenderRequest) -> str:
    focus_ratio = estimate_focus_ratio(render_request.tracking_boxes)
    subtitle_filter = f"subtitles='{escape_filter_path(Path(render_request.subtitle_path))}'"
    if render_request.crop_mode == "contain":
        framing = build_contain_pad_filter()
    else:
        framing = ",".join(
            [
                build_vertical_crop_filter(focus_ratio),
                "scale=1188:2112",
                "crop=1080:1920:(iw-ow)/2:(ih-oh)/2",
            ]
        )
    return ",".join(
        [
            "fps=30",
            framing,
            "eq=contrast=1.03:saturation=1.08",
            subtitle_filter,
        ]
    )


def build_ffmpeg_command(settings: Settings, render_request: RenderRequest) -> list[str]:
    start_seconds = render_request.start_ms / 1000.0
    duration_seconds = max(1.0, (render_request.end_ms - render_request.start_ms) / 1000.0)
    return [
        settings.ffmpeg_binary,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        render_request.input_video_path,
        "-t",
        f"{duration_seconds:.3f}",
        "-vf",
        build_filter_chain(render_request),
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        render_request.output_path,
    ]


def render_clip(settings: Settings, render_request: RenderRequest) -> list[str]:
    Path(render_request.output_path).parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(settings, render_request)
    subprocess.run(command, check=True, capture_output=True, text=True)
    return command
