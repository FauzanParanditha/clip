from __future__ import annotations

import json
from pathlib import Path

from .contracts import TrackingBox


def source_uses_av1(metadata_path: str | None) -> bool:
    if not metadata_path:
        return False

    path = Path(metadata_path)
    if not path.exists():
        return False

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    codecs: list[str] = []
    for key in ("vcodec",):
        value = metadata.get(key)
        if isinstance(value, str):
            codecs.append(value)

    for collection_key in ("requested_formats", "requested_downloads", "formats"):
        collection = metadata.get(collection_key)
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            value = item.get("vcodec")
            if isinstance(value, str):
                codecs.append(value)

    normalized = " ".join(codecs).lower()
    return "av01" in normalized or normalized.strip() == "av1" or " av1" in normalized


def detect_subject_tracking(
    video_path: str,
    start_ms: int,
    end_ms: int,
    sample_step_ms: int = 1200,
    metadata_path: str | None = None,
) -> list[TrackingBox]:
    # OpenCV emits repeated AV1 decode warnings on CPU-only hosts. In that case,
    # fall back to center crop and skip tracking entirely.
    if source_uses_av1(metadata_path):
        return []

    try:
        import cv2  # type: ignore
    except ImportError:
        return []

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detections: list[TrackingBox] = []

    sample_points = range(start_ms, end_ms, sample_step_ms)
    for timestamp_ms in sample_points:
        frame_index = int((timestamp_ms / 1000.0) * fps)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            continue
        x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
        detections.append(
            TrackingBox(
                frame_ms=timestamp_ms - start_ms,
                frame_width=frame_width,
                frame_height=frame_height,
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
            )
        )

    capture.release()
    return detections
