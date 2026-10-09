"""Per-page chunking.

Chunks never span pages, so every chunk has exactly one unambiguous page number.
"""
import re

from .config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
from .models import Chunk, PageText

_SENTENCE_BREAKS = (". ", "! ", "? ", "; ", "\n")


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Sliding window over an oversized paragraph, preferring sentence breaks."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max((window.rfind(b) for b in _SENTENCE_BREAKS), default=-1)
            if cut > len(window) - 200 and cut > 0:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _split_page(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]

    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(para) <= max_chars:
            current = para
        else:
            pieces.extend(_split_long_text(para, max_chars, overlap))
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def chunk_pages(pages: list[PageText], doc_id: str, filename: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in _split_page(page.text):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{index}",
                    doc_id=doc_id,
                    filename=filename,
                    page=page.page_number,
                    index=index,
                    text=piece,
                )
            )
            index += 1
    return chunks
