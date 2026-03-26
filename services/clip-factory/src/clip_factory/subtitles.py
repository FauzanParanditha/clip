from __future__ import annotations

from dataclasses import dataclass

from .contracts import TranscriptWord


@dataclass(slots=True)
class SubtitleChunk:
    words: list[TranscriptWord]

    @property
    def start_ms(self) -> int:
        return self.words[0].start_ms

    @property
    def end_ms(self) -> int:
        return self.words[-1].end_ms


def ms_to_ass(milliseconds: int) -> str:
    centiseconds = max(0, milliseconds) // 10
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def split_lines(words: list[TranscriptWord], max_chars_per_line: int = 22) -> tuple[list[TranscriptWord], list[TranscriptWord]]:
    if len(words) <= 3:
        return words, []

    best_index = len(words) // 2
    best_score = 10**9
    for index in range(1, len(words)):
        left = " ".join(word.text for word in words[:index])
        right = " ".join(word.text for word in words[index:])
        score = max(len(left), len(right))
        if score < best_score and len(left) <= max_chars_per_line * 2 and len(right) <= max_chars_per_line * 2:
            best_score = score
            best_index = index
    return words[:best_index], words[best_index:]


def group_words(words: list[TranscriptWord], max_words: int = 8, max_gap_ms: int = 650) -> list[SubtitleChunk]:
    chunks: list[SubtitleChunk] = []
    buffer: list[TranscriptWord] = []
    for index, word in enumerate(words):
        buffer.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        gap_ms = (next_word.start_ms - word.end_ms) if next_word else 0
        if len(buffer) >= max_words or gap_ms > max_gap_ms or word.text.endswith((".", "?", "!")):
            chunks.append(SubtitleChunk(buffer[:]))
            buffer = []
    if buffer:
        chunks.append(SubtitleChunk(buffer[:]))
    return chunks


def _soft_highlight(text: str) -> str:
    return f"{{\\c&H90F5FF&}}{text}{{\\c&HFFFFFF&}}"


def _active_highlight(text: str) -> str:
    return f"{{\\c&H2BFFFF&\\b1}}{text}{{\\b0\\c&HFFFFFF&}}"


def _should_soft_highlight(word: TranscriptWord, keywords: set[str]) -> bool:
    cleaned = "".join(char for char in word.text.lower() if char.isalnum())
    if not cleaned:
        return False
    return cleaned in keywords or any(char.isdigit() for char in cleaned) or word.text.istitle()


def render_chunk_text(chunk: SubtitleChunk, active_index: int, keywords: set[str]) -> str:
    rendered_words = []
    for index, word in enumerate(chunk.words):
        token = word.text
        if index == active_index:
            rendered_words.append(_active_highlight(token))
        elif _should_soft_highlight(word, keywords):
            rendered_words.append(_soft_highlight(token))
        else:
            rendered_words.append(token)
    left, right = split_lines(chunk.words)
    if not right:
        return " ".join(rendered_words)
    split_at = len(left)
    return " ".join(rendered_words[:split_at]) + r"\N" + " ".join(rendered_words[split_at:])


def render_ass(words: list[TranscriptWord], hook_text: str, keywords: list[str], font_name: str = "Arial") -> str:
    chunks = group_words(words)
    keyword_set = {keyword.lower() for keyword in keywords}
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: Clip,{font_name},68,&H00FFFFFF,&H00FFFFFF,&H00111111,&H32000000,1,0,0,0,100,100,0,0,1,3,0,2,80,80,210,1",
        f"Style: Hook,{font_name},54,&H002BFFFF,&H002BFFFF,&H00111111,&H26000000,1,0,0,0,100,100,0,0,1,2,0,8,90,90,1400,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    if hook_text:
        lines.append(
            f"Dialogue: 0,{ms_to_ass(0)},{ms_to_ass(min(words[-1].end_ms, 1600))},Hook,,0,0,0,,{hook_text.replace(',', ' ')}"
        )

    for chunk in chunks:
        for word_index, word in enumerate(chunk.words):
            start_ms = word.start_ms
            if word_index + 1 < len(chunk.words):
                end_ms = chunk.words[word_index + 1].start_ms
            else:
                end_ms = chunk.end_ms + 80
            text = render_chunk_text(chunk, word_index, keyword_set)
            lines.append(
                f"Dialogue: 0,{ms_to_ass(start_ms)},{ms_to_ass(end_ms)},Clip,,0,0,0,,{text}"
            )

    return "\n".join(lines) + "\n"

