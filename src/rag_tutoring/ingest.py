"""Load PDFs and split them into overlapping, page-tagged chunks.

The output is a list of :class:`Chunk` objects -- plain text plus the metadata
needed to cite it (document title, page number). Nothing here knows about
embeddings or the vector store; that separation is what keeps each concern
testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from rag_tutoring.config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus everything needed to cite it."""

    text: str
    source: str  # document title (PDF filename stem), shown in citations
    source_type: str  # "paper" | "textbook"
    page: int  # 1-indexed page the chunk was extracted from
    chunk_index: int  # position of this chunk within its page

    @property
    def id(self) -> str:
        """Stable id, so re-ingesting a document upserts instead of duplicating."""
        return f"{self.source}::p{self.page:04d}::c{self.chunk_index:02d}"


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` for each page that has extractable text.

    Page numbers are 1-indexed to match how a reader cites them. Pages whose
    extracted text is empty (e.g. a full-page figure) are skipped rather than
    emitted as blank chunks.
    """
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split text into word windows of ``chunk_words`` with ``overlap_words`` overlap."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break  # this window already reached the end; a further one would be redundant
    return chunks


def chunk_pdf(
    path: Path,
    source_type: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Load a PDF and chunk it page by page.

    Chunking never crosses a page boundary, so every chunk cites exactly one
    page. The tradeoff: a passage split across a page break lands in two chunks
    with no overlap bridging them. Acceptable for Phase 1; revisit if recall
    suffers on concepts that straddle pages.
    """
    source = path.stem
    chunks: list[Chunk] = []
    for page_number, text in load_pdf(path):
        for idx, piece in enumerate(chunk_text(text, chunk_words, overlap_words)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    source_type=source_type,
                    page=page_number,
                    chunk_index=idx,
                )
            )
    return chunks
