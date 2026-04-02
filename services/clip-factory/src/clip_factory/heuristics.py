from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .contracts import ClipMetadata, SegmentCandidate, SourceAsset, TranscriptDocument, TranscriptWord


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "we",
    "will",
    "with",
    "you",
    "your",
}
HOOK_WORDS = {
    "actually",
    "biggest",
    "breakdown",
    "crazy",
    "critical",
    "dangerous",
    "important",
    "mistake",
    "never",
    "news",
    "secret",
    "truth",
    "warning",
    "why",
}
INTRO_WORDS = {"welcome", "today", "episode", "podcast", "before", "start", "joining"}
OUTRO_WORDS = {"subscribe", "watching", "next time", "like and subscribe", "thanks for watching"}
SPONSOR_WORDS = {"sponsor", "sponsored", "promo", "discount", "code", "partnered"}
FILLER_WORDS = {"basically", "literally", "actually", "kind of", "sort of", "you know", "i mean"}
NEWS_TITLE_STOPWORDS = STOPWORDS | {
    "after",
    "amid",
    "announces",
    "announcement",
    "bbc",
    "breaking",
    "channel",
    "exclusive",
    "headline",
    "inews",
    "latest",
    "new",
    "news",
    "official",
    "officials",
    "report",
    "reported",
    "reports",
    "says",
    "said",
    "ten",
    "terkini",
    "today",
    "tonight",
    "update",
}
NEWS_SIDE_ANGLE_WORDS = {
    "audience",
    "believer",
    "believers",
    "community",
    "congregation",
    "critic",
    "critics",
    "opinion",
    "people",
    "public",
    "reaction",
    "reactions",
    "resident",
    "residents",
    "sentiment",
    "supporter",
    "supporters",
    "viewer",
    "viewers",
    "voter",
    "voters",
}


@dataclass(slots=True)
class Sentence:
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    words: list[TranscriptWord]


def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def format_timestamp(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def slice_words(words: list[TranscriptWord], start_ms: int, end_ms: int, normalize: bool = False) -> list[TranscriptWord]:
    selected = []
    for word in words:
        if word.end_ms < start_ms or word.start_ms > end_ms:
            continue
        if normalize:
            selected.append(
                TranscriptWord(
                    text=word.text,
                    start_ms=max(0, word.start_ms - start_ms),
                    end_ms=max(0, word.end_ms - start_ms),
                    confidence=word.confidence,
                    speaker=word.speaker,
                )
            )
        else:
            selected.append(word)
    return selected


def split_sentences(words: list[TranscriptWord]) -> list[Sentence]:
    sentences: list[Sentence] = []
    buffer: list[TranscriptWord] = []
    for index, word in enumerate(words):
        buffer.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        gap_ms = (next_word.start_ms - word.end_ms) if next_word else 0
        if word.text.endswith((".", "?", "!")) or gap_ms >= 1100 or len(buffer) >= 28:
            text = " ".join(item.text for item in buffer).strip()
            confidence = sum(item.confidence for item in buffer) / max(1, len(buffer))
            sentences.append(
                Sentence(
                    text=text,
                    start_ms=buffer[0].start_ms,
                    end_ms=buffer[-1].end_ms,
                    confidence=confidence,
                    words=buffer[:],
                )
            )
            buffer = []
    if buffer:
        text = " ".join(item.text for item in buffer).strip()
        confidence = sum(item.confidence for item in buffer) / max(1, len(buffer))
        sentences.append(
            Sentence(
                text=text,
                start_ms=buffer[0].start_ms,
                end_ms=buffer[-1].end_ms,
                confidence=confidence,
                words=buffer[:],
            )
        )
    return sentences


def _content_tokens(text: str) -> list[str]:
    tokens = [normalize_token(token) for token in text.split()]
    return [token for token in tokens if token and token not in STOPWORDS]


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    tokens = _content_tokens(text)
    counts = Counter(token for token in tokens if len(token) > 2)
    return [token for token, _ in counts.most_common(limit)]


def _headline_tokens(text: str) -> list[str]:
    return [
        token
        for token in _content_tokens(text)
        if len(token) > 2 and token not in NEWS_TITLE_STOPWORDS
    ]


def _headline_phrases(text: str) -> set[str]:
    tokens = _headline_tokens(text)
    phrases: set[str] = set()
    for size in (2, 3):
        for index in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[index : index + size])
            if len(phrase.replace(" ", "")) >= 8:
                phrases.add(phrase)
    return phrases


def _news_alignment_metrics(text: str, source_title: str) -> tuple[int, float, int, int, float]:
    title_tokens = _headline_tokens(source_title)
    if not title_tokens:
        return 0, 0.0, 0, 0, 0.0

    candidate_tokens = _content_tokens(text)
    if not candidate_tokens:
        return 0, 0.0, 0, 0, 0.0

    title_set = set(title_tokens)
    candidate_set = set(candidate_tokens)
    overlap_count = len(title_set & candidate_set)
    overlap_ratio = overlap_count / len(title_set)
    opener_overlap = len(title_set & set(candidate_tokens[:14]))
    phrase_hits = sum(1 for phrase in _headline_phrases(source_title) if phrase in text.lower())
    anchor_overlap = min(
        1.0,
        (overlap_count / max(2.0, min(6.0, len(title_set))))
        + min(0.45, opener_overlap * 0.12)
        + min(0.45, phrase_hits * 0.22),
    )
    return overlap_count, overlap_ratio, opener_overlap, phrase_hits, anchor_overlap


def _penalty_hits(text: str, phrases: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for phrase in phrases if phrase in lower)


def build_candidate_segments(
    transcript: TranscriptDocument,
    target_count: int = 8,
    *,
    content_type: str = "podcast",
    source_title: str = "",
) -> list[SegmentCandidate]:
    sentences = split_sentences(transcript.words)
    candidates: list[SegmentCandidate] = []
    min_duration = 20_000
    max_duration = 75_000
    preferred_duration = 42_000

    for start_index in range(len(sentences)):
        window_words: list[TranscriptWord] = []
        for end_index in range(start_index, len(sentences)):
            window_words.extend(sentences[end_index].words)
            start_ms = sentences[start_index].start_ms
            end_ms = sentences[end_index].end_ms
            duration = end_ms - start_ms
            if duration < min_duration:
                continue
            if duration > max_duration:
                break

            text = " ".join(word.text for word in window_words).strip()
            confidence = sum(word.confidence for word in window_words) / max(1, len(window_words))
            score, reason, flags = score_segment(
                text,
                duration,
                confidence,
                start_ms,
                transcript.words[-1].end_ms,
                content_type=content_type,
                source_title=source_title,
            )
            hook_text = extract_hook_text(text)
            candidate = SegmentCandidate(
                segment_id=f"seg_{hashlib.sha1(f'{start_ms}:{end_ms}'.encode('utf-8')).hexdigest()[:10]}",
                start_ms=start_ms,
                end_ms=end_ms,
                score=round(score, 2),
                reason=reason,
                hook_text=hook_text,
                keywords=extract_keywords(text),
                confidence=round(confidence, 3),
                text=text,
                flags=flags,
            )
            candidates.append(candidate)

            if duration >= preferred_duration and len(candidates) >= target_count * 6:
                break

    deduped: dict[tuple[int, int], SegmentCandidate] = {}
    for candidate in candidates:
        key = (candidate.start_ms // 5000, candidate.end_ms // 5000)
        existing = deduped.get(key)
        if existing is None or existing.score < candidate.score:
            deduped[key] = candidate
    return sorted(deduped.values(), key=lambda item: item.score, reverse=True)


def extract_hook_text(text: str, limit: int = 90) -> str:
    sanitized = re.sub(r"\s+", " ", text).strip()
    if len(sanitized) <= limit:
        return sanitized
    cut = sanitized[:limit].rsplit(" ", 1)[0].strip()
    return cut or sanitized[:limit].strip()


def score_segment(
    text: str,
    duration_ms: int,
    confidence: float,
    start_ms: int,
    total_duration_ms: int,
    *,
    content_type: str = "podcast",
    source_title: str = "",
) -> tuple[float, str, list[str]]:
    lower = text.lower()
    content_tokens = _content_tokens(text)
    token_count = len(content_tokens)
    unique_ratio = len(set(content_tokens)) / max(1, token_count)
    punctuation_burst = lower.count("?") + lower.count("!")
    numeric_density = sum(1 for token in text.split() if any(char.isdigit() for char in token))
    hook_hits = sum(1 for token in content_tokens[:12] if token in HOOK_WORDS)
    intro_penalty = _penalty_hits(lower, INTRO_WORDS) if start_ms < 90_000 else 0
    outro_penalty = _penalty_hits(lower, OUTRO_WORDS) if total_duration_ms - start_ms < 120_000 else _penalty_hits(lower, OUTRO_WORDS)
    sponsor_penalty = _penalty_hits(lower, SPONSOR_WORDS)
    filler_penalty = _penalty_hits(lower, FILLER_WORDS)
    duration_seconds = duration_ms / 1000.0
    duration_score = max(0.0, 1.0 - abs(duration_seconds - 45.0) / 40.0)
    info_density = min(1.0, token_count / max(15.0, duration_seconds))
    standalone = min(1.0, unique_ratio + (0.08 if text.endswith((".", "?", "!")) else 0.0))
    novelty = min(1.0, unique_ratio)
    quote_density = min(1.0, (punctuation_burst * 0.12) + (0.08 if '"' in text else 0.0))
    speaker_energy = min(1.0, 0.25 + punctuation_burst * 0.18 + numeric_density * 0.05)
    confidence_score = min(1.0, max(0.0, confidence))
    news_headline_alignment = 0.0
    news_alignment_penalty = 0.0
    news_side_angle_penalty = 0.0
    news_side_angle_hits = 0
    headline_overlap_count = 0
    headline_phrase_hits = 0

    weighted = (
        duration_score * 0.16
        + min(1.0, hook_hits * 0.25 + punctuation_burst * 0.05) * 0.23
        + standalone * 0.18
        + novelty * 0.12
        + quote_density * 0.07
        + speaker_energy * 0.10
        + info_density * 0.09
        + confidence_score * 0.05
    )

    if content_type == "news" and source_title:
        headline_overlap_count, headline_overlap_ratio, opener_overlap, headline_phrase_hits, news_headline_alignment = (
            _news_alignment_metrics(text, source_title)
        )
        news_side_angle_hits = _penalty_hits(lower, NEWS_SIDE_ANGLE_WORDS)
        weighted += news_headline_alignment * 0.24
        if start_ms <= int(total_duration_ms * 0.38) and headline_overlap_count >= 2:
            weighted += 0.05
        if headline_overlap_count == 0:
            news_alignment_penalty += 0.16
        elif headline_overlap_ratio < 0.18 and headline_phrase_hits == 0:
            news_alignment_penalty += 0.08
        if news_side_angle_hits and headline_overlap_count < 2:
            news_side_angle_penalty += 0.08 * min(news_side_angle_hits, 2)
        if start_ms >= int(total_duration_ms * 0.7) and headline_overlap_count < 2 and news_side_angle_hits:
            news_side_angle_penalty += 0.05

    total_penalty = (
        sponsor_penalty * 0.18
        + filler_penalty * 0.05
        + intro_penalty * 0.10
        + outro_penalty * 0.12
        + news_alignment_penalty
        + news_side_angle_penalty
    )
    raw_score = max(0.0, min(1.0, weighted - total_penalty))
    score = raw_score * 100

    positive_signals = []
    if content_type == "news" and (headline_overlap_count >= 2 or headline_phrase_hits):
        positive_signals.append("headline aligned")
    if hook_hits:
        positive_signals.append("strong opener")
    if info_density > 0.72:
        positive_signals.append("high info density")
    if speaker_energy > 0.45:
        positive_signals.append("good delivery energy")
    if novelty > 0.72:
        positive_signals.append("low repetition")
    if not positive_signals:
        positive_signals.append("clean standalone segment")

    penalties = []
    flags: list[str] = []
    if sponsor_penalty:
        penalties.append("sponsor language")
        flags.append("sponsor_risk")
    if intro_penalty:
        penalties.append("intro framing")
        flags.append("intro_risk")
    if outro_penalty:
        penalties.append("outro framing")
        flags.append("outro_risk")
    if filler_penalty >= 2:
        penalties.append("too much filler")
        flags.append("filler_risk")
    if confidence < 0.72:
        penalties.append("low transcript confidence")
        flags.append("low_confidence")
    if content_type == "news" and news_alignment_penalty >= 0.08:
        penalties.append("weak headline alignment")
        flags.append("headline_miss")
    if content_type == "news" and news_side_angle_penalty >= 0.08:
        penalties.append("side-angle tangent")
        flags.append("side_angle")

    reason = ", ".join(positive_signals[:2])
    if penalties:
        reason = f"{reason}; penalties: {', '.join(penalties[:2])}"
    return score, reason, flags


def _news_headline_signature(candidate: SegmentCandidate, source_title: str) -> frozenset[str]:
    title_tokens = set(_headline_tokens(source_title))
    if not title_tokens:
        return frozenset()
    signature = {token for token in candidate.keywords if token in title_tokens}
    if len(signature) < 2:
        return frozenset()
    return frozenset(signature)


def select_segments(
    candidates: list[SegmentCandidate],
    requested_count: int,
    transcript_confidence: float,
    *,
    content_type: str = "podcast",
    source_title: str = "",
) -> tuple[list[SegmentCandidate], bool]:
    target = max(5, min(12, requested_count))
    review_needed = transcript_confidence < 0.76
    if review_needed:
        target = max(3, target - 2)

    selected: list[SegmentCandidate] = []
    news_signature_cache: list[frozenset[str]] = []
    similarity_threshold = 0.5 if content_type == "news" else 0.66
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(overlaps(candidate, existing) for existing in selected):
            continue
        if any(keyword_similarity(candidate, existing) > similarity_threshold for existing in selected):
            continue
        if content_type == "news" and source_title:
            signature = _news_headline_signature(candidate, source_title)
            if signature and any(signature <= existing or existing <= signature for existing in news_signature_cache if existing):
                continue
        selected.append(candidate)
        if content_type == "news" and source_title:
            news_signature_cache.append(_news_headline_signature(candidate, source_title))
        if len(selected) >= target:
            break

    if len(selected) < 5 and len(selected) < len(candidates):
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if candidate in selected:
                continue
            if any(overlaps(candidate, existing) for existing in selected):
                continue
            if any(abs(candidate.start_ms - existing.start_ms) < 12_000 for existing in selected):
                continue
            if any(keyword_similarity(candidate, existing) > similarity_threshold for existing in selected):
                continue
            if content_type == "news" and source_title:
                signature = _news_headline_signature(candidate, source_title)
                if signature and any(signature <= existing or existing <= signature for existing in news_signature_cache if existing):
                    continue
            selected.append(candidate)
            if content_type == "news" and source_title:
                news_signature_cache.append(_news_headline_signature(candidate, source_title))
            if len(selected) >= min(5, target):
                break

    return selected, review_needed


def overlaps(left: SegmentCandidate, right: SegmentCandidate, gap_ms: int = 5_000) -> bool:
    return not (left.end_ms + gap_ms <= right.start_ms or right.end_ms + gap_ms <= left.start_ms)


def keyword_similarity(left: SegmentCandidate, right: SegmentCandidate) -> float:
    left_set = set(left.keywords)
    right_set = set(right.keywords)
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union)


def generate_clip_metadata(candidate: SegmentCandidate, source: SourceAsset) -> ClipMetadata:
    primary = candidate.keywords[0] if candidate.keywords else "topic"
    hook = candidate.hook_text.rstrip(".")
    titles = [
        hook[:90],
        f"{primary.title()} in under a minute",
        f"The {primary} moment people replay",
    ]
    titles = [title for title in titles if title]
    hashtags = [f"#{keyword.replace(' ', '')}" for keyword in candidate.keywords[:8]]
    hashtags.extend(["#shorts", "#tiktokclips", f"#{source.uploader.replace(' ', '')}"] if source.uploader else ["#shorts", "#tiktokclips"])
    deduped_hashtags = []
    for hashtag in hashtags:
        if hashtag.lower() not in {item.lower() for item in deduped_hashtags}:
            deduped_hashtags.append(hashtag)
    caption = f"{hook}. Source: {source.title or source.source_url}"
    return ClipMetadata(
        hook_text=hook[:90],
        titles=titles[:3],
        caption=caption[:220],
        hashtags=deduped_hashtags[:15],
        highlight_keywords=candidate.keywords[:6],
        source_timestamp_label=format_timestamp(candidate.start_ms),
        selection_reason=candidate.reason,
    )
