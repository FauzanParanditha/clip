from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .contracts import TranscriptDocument, TranscriptWord


def run_transcription(audio_path: str, settings: Settings) -> TranscriptDocument:
    sidecar = Path(audio_path).with_suffix(".transcript.json")
    if sidecar.exists():
        return TranscriptDocument.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Transcription engine unavailable. Install faster-whisper or provide a sidecar transcript.") from exc

    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    segments, info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
    words: list[TranscriptWord] = []
    confidence_values: list[float] = []

    for segment in segments:
        for word in segment.words or []:
            confidence = float(getattr(word, "probability", 0.85))
            confidence_values.append(confidence)
            words.append(
                TranscriptWord(
                    text=(word.word or "").strip(),
                    start_ms=int(float(word.start) * 1000),
                    end_ms=int(float(word.end) * 1000),
                    confidence=confidence,
                )
            )

    if not words:
        raise RuntimeError("Whisper returned no word-level transcript")

    average_confidence = sum(confidence_values) / max(1, len(confidence_values))
    return TranscriptDocument(
        language=getattr(info, "language", None),
        average_confidence=average_confidence,
        words=words,
    )
