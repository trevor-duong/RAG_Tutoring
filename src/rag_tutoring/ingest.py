"""Load PDFs and split them into overlapping, page-tagged chunks.

The output is a list of :class:`Chunk` objects -- plain text plus the metadata
needed to cite it (document title, page number). Nothing here knows about
embeddings or the vector store; that separation is what keeps each concern
testable on its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from rag_tutoring.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, EMBEDDING_MODEL


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


# PDF text extraction leaves two kinds of junk that otherwise survive into chunk
# text, and both are invisible when you eyeball a snippet.
#
# Lone surrogates -- half a surrogate pair for a character outside the BMP,
# usually a mathematical alphanumeric symbol (U+1D4xx). The string is then not
# valid UTF-8 and the tokenizer rejects it outright with a TypeError.
#
# Control characters -- glyphs from fonts with custom encodings that decode to
# C0/C1 code points rather than the symbol they draw: the circled plus in
# ResNet's "F(x) + x", checkmarks in results tables. They render as nothing in a
# quoted citation, they embed as [UNK], and U+0000 is hostile to the storage
# layer. Measured across the corpus, ~19% of chunks carried at least one.
#
# Both are replaced with a space rather than deleted. Several sit between a
# label and a number with no surrounding whitespace, where deleting would fuse
# them into a value that was never in the source ("Top-5 <ctrl> 47.0" would
# become "Top-547.0"). Tab, newline and CR are left alone -- ``str.split()``
# handles them.
_EXTRACTION_JUNK = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]")


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` for each page that has extractable text.

    Page numbers are 1-indexed to match how a reader cites them. Pages whose
    extracted text is empty (e.g. a full-page figure) are skipped rather than
    emitted as blank chunks.
    """
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _EXTRACTION_JUNK.sub(" ", page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


@lru_cache(maxsize=4)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer  # heavy import, and only the vocab is needed

    return AutoTokenizer.from_pretrained(model_name)


def token_counter(model_name: str = EMBEDDING_MODEL) -> Callable[[str], int]:
    """Build a memoised ``word -> word-piece count`` function for a model's vocabulary.

    Chunking needs to respect the embedding model's input limit, but it does
    not need the model -- only the cost of each word. Passing that cost function
    in (rather than importing a tokenizer here) keeps this module independent of
    which embedding model is in use, the same inversion that keeps ``store.py``
    swappable. Tests can supply a trivial counter instead.
    """
    tokenizer = _tokenizer(model_name)
    cache: dict[str, int] = {}

    def count(word: str) -> int:
        cost = cache.get(word)
        if cost is None:
            cost = len(tokenizer.encode(word, add_special_tokens=False))
            cache[word] = cost
        return cost

    return count


def pack_words(costs: list[int], max_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Group word indices into ``[start, end)`` windows that fit the token budget.

    Greedy: fill a window until the next word would exceed ``max_tokens``, then
    step back far enough to carry ``overlap_tokens`` of context into the next
    one. Works on token *costs* rather than the words themselves because the two
    callers (chunking and tests) only need the boundaries.

    Relies on per-word costs summing to the cost of the joined string, which
    holds for the word-piece tokenizers used here: they pre-split on whitespace
    before splitting into subwords, so no token ever straddles a space.
    """
    spans: list[tuple[int, int]] = []
    n = len(costs)
    start = 0
    while start < n:
        end, total = start, 0
        while end < n and total + costs[end] <= max_tokens:
            total += costs[end]
            end += 1
        if end == start:
            # One word blows the whole budget (a mis-extracted table, say).
            # Emit it alone -- it will be truncated, but the packer must advance.
            end = start + 1
        spans.append((start, end))
        if end >= n:
            break
        carried, next_start = 0, end
        while next_start > start + 1 and carried + costs[next_start - 1] <= overlap_tokens:
            next_start -= 1
            carried += costs[next_start]
        start = max(next_start, start + 1)  # never stall, even on costly words
    return spans


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping windows that each fit the embedding window.

    Chunks hold the *original* words. Rebuilding them by decoding token ids
    would be lossy -- this tokenizer is uncased and splits punctuation, so
    "Multi-Head Attention (Vaswani et al., 2017)" decodes back as
    "multi - head attention ( vaswani et al., 2017 )". Chunk text is quoted to
    students as a citation, so it has to survive verbatim.
    """
    words = text.split()
    if not words:
        return []
    count_tokens = count_tokens or token_counter()
    costs = [count_tokens(w) for w in words]
    return [" ".join(words[s:e]) for s, e in pack_words(costs, max_tokens, overlap_tokens)]


def chunk_pdf(
    path: Path,
    source_type: str,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Load a PDF and chunk it page by page.

    Chunking never crosses a page boundary, so every chunk cites exactly one
    page. The tradeoff: a passage split across a page break lands in two chunks
    with no overlap bridging them. Acceptable for Phase 1; revisit if recall
    suffers on concepts that straddle pages.

    Papers and textbooks share these settings deliberately: the budget is set by
    the embedding model's input window, which does not care what kind of
    document the text came from.
    """
    source = path.stem
    count_tokens = count_tokens or token_counter()  # built once, reused across pages
    chunks: list[Chunk] = []
    for page_number, text in load_pdf(path):
        for idx, piece in enumerate(chunk_text(text, count_tokens, max_tokens, overlap_tokens)):
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
