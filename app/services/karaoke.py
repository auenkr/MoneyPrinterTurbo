"""Short-form "karaoke" captions: a few words at a time, active word highlighted.

This is the caption style used by most high-retention YouTube Shorts / TikTok
videos. Input is word-level subtitle cues ``[((start, end), text), ...]`` and
output is a list of positioned MoviePy ``ImageClip`` objects that cover the
narration without gaps or flicker.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
from moviepy import ImageClip
from PIL import Image, ImageDraw, ImageFont

SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")
CLAUSE_END = (",", ";", ":", "，", "；", "：")


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Style:
    font_path: str
    font_size: int
    color: str = "#FFFFFF"
    highlight_color: str = "#FFE600"
    stroke_color: str = "#000000"
    stroke_width: int = 8
    uppercase: bool = True
    max_words: int = 3
    max_width: int = 960
    shadow: bool = True


def _is_wide(text: str) -> bool:
    return any(unicodedata.east_asian_width(ch) in ("W", "F") for ch in text)


def _join_gap(a: str, b: str) -> str:
    """No space between two CJK tokens; a space everywhere else."""
    return "" if (_is_wide(a[-1:]) and _is_wide(b[:1])) else " "


def _line(tokens: list[str]) -> str:
    out = tokens[0] if tokens else ""
    for prev, tok in zip(tokens, tokens[1:]):
        out += _join_gap(prev, tok) + tok
    return out


def _display(text: str, style: Style) -> str:
    text = text.strip()
    return text.upper() if style.uppercase else text


def chunk_words(words: list[Word], style: Style, font: ImageFont.FreeTypeFont,
                max_gap: float = 0.3) -> list[list[Word]]:
    """Group words into short on-screen phrases that fit on one line."""
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if not word.text.strip():
            continue
        if current:
            prev = current[-1]
            candidate = current + [word]
            line = _line([_display(w.text, style) for w in candidate])
            # CJK cues are per character, so allow more of them per phrase.
            limit = style.max_words * 3 if _is_wide(word.text) else style.max_words
            too_wide = font.getlength(line) + 2 * style.stroke_width > style.max_width
            boundary = (
                len(current) >= limit
                or too_wide
                or word.start - prev.end > max_gap
                or prev.text.rstrip().endswith(SENTENCE_END)
                or (prev.text.rstrip().endswith(CLAUSE_END) and len(current) >= 2)
            )
            if boundary:
                chunks.append(current)
                current = []
        current.append(word)
    if current:
        chunks.append(current)
    return chunks


def _render_chunk(chunk: list[Word], active: int, style: Style,
                  font: ImageFont.FreeTypeFont) -> np.ndarray:
    tokens = [_display(w.text, style) for w in chunk]
    # A single long word may still exceed the safe width: shrink to fit.
    full = font.getlength(_line(tokens)) + 2 * style.stroke_width
    if full > style.max_width:
        size = max(12, int(style.font_size * style.max_width / full))
        font = ImageFont.truetype(style.font_path, size)
    sw = style.stroke_width
    gaps = [font.getlength(_join_gap(a, b)) if _join_gap(a, b) else 0
            for a, b in zip(tokens, tokens[1:])] + [0]
    widths = [font.getlength(t) for t in tokens]
    ascent, descent = font.getmetrics()
    shadow = max(2, style.font_size // 18) if style.shadow else 0
    pad = sw + shadow + 4
    width = int(sum(widths) + sum(gaps) + 2 * pad)
    height = int(ascent + descent + 2 * pad)
    img = Image.new("RGBA", (max(width, 1), max(height, 1)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = float(pad)
    for i, token in enumerate(tokens):
        fill = style.highlight_color if i == active else style.color
        if shadow:
            draw.text((x + shadow, pad + shadow), token, font=font,
                      fill=(0, 0, 0, 150), stroke_width=sw, stroke_fill=(0, 0, 0, 150))
        draw.text((x, pad), token, font=font, fill=fill,
                  stroke_width=sw, stroke_fill=style.stroke_color)
        x += widths[i] + gaps[i]
    return np.array(img)


def build_clips(cues, style: Style, video_width: int, video_height: int,
                y_ratio: float = 0.68, total_duration: float | None = None):
    """Return positioned ImageClips for word-level cues.

    ``y_ratio`` is the vertical centre of the caption as a fraction of height;
    0.68 keeps it above the Shorts UI overlay at the bottom of the screen.
    """
    words = sorted(
        (Word(float(t[0]), float(t[1]), str(text)) for t, text in cues),
        key=lambda w: w.start,
    )
    if not words:
        return []
    font = ImageFont.truetype(style.font_path, style.font_size)
    chunks = chunk_words(words, style, font)
    clips = []
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else None
        # Hold the phrase until the next one starts (short pauses) but never
        # leave stale text on screen through long silences.
        chunk_end = chunk[-1].end + 0.25
        if next_start is not None and next_start - chunk[-1].end <= 0.6:
            chunk_end = next_start
        elif next_start is not None:
            chunk_end = min(chunk_end, next_start)
        if total_duration is not None:
            chunk_end = min(chunk_end, total_duration)
        # Never drop the last word because of overlapping cues.
        chunk_end = max(chunk_end, chunk[-1].start + 0.1)
        for wi, word in enumerate(chunk):
            start = word.start
            end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
            if total_duration is not None:
                end = min(end, total_duration)
            if end - start <= 0.01:
                continue
            frame = _render_chunk(chunk, wi, style, font)
            h, w = frame.shape[:2]
            y = int(video_height * y_ratio - h / 2)
            y = max(0, min(y, video_height - h))
            clip = (
                ImageClip(frame, transparent=True)
                .with_start(start)
                .with_duration(end - start)
                .with_position(((video_width - w) // 2, y))
            )
            clips.append(clip)
    return clips
