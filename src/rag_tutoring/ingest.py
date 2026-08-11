"""Load PDFs and split them into overlapping, page-tagged chunks.

The output is a list of :class:`Chunk` objects -- plain text plus the metadata
needed to cite it (document title, page number). Nothing here knows about
embeddings or the vector store; that separation is what keeps each concern
testable on its own.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from rag_tutoring.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    DATA_RAW,
    EMBEDDING_MODEL,
    PAGES_CACHE,
)
from rag_tutoring.structure import is_structural

# Source types, each stored in ``data/raw/<type>s/``. The type rides along on
# every chunk so retrieval can distinguish a paper's claim from a textbook's
# explanation -- they answer a student's question differently.
SOURCE_TYPES = ("paper", "textbook")


def corpus_documents(data_raw: Path = DATA_RAW) -> list[tuple[Path, str]]:
    """Every source PDF with its source type, in a stable (sorted) order.

    One definition of "what the corpus is", shared by ingestion, the eval
    harness, and any audit script -- so a document cannot be indexed but missing
    from an audit, or vice versa.
    """
    return [
        (path, kind)
        for kind in SOURCE_TYPES
        for path in sorted((data_raw / f"{kind}s").glob("*.pdf"))
    ]


@dataclass(frozen=True)
class Page:
    """One extracted page, as cached between runs.

    ``page`` is pypdf's page index (1-based), the same number :class:`Chunk`
    carries -- not the number printed on the page. Anything deriving eval ground
    truth has to use this numbering or every label lands on the wrong page.
    """

    source: str
    source_type: str
    page: int
    text: str


def write_pages(path: Path, pages: Iterable[Page]) -> int:
    """Write the extraction cache as JSONL, returning how many pages were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w") as handle:
        for p in pages:
            handle.write(json.dumps(p.__dict__) + "\n")
            written += 1
    return written


def read_pages(path: Path = PAGES_CACHE) -> list[Page]:
    """Read the extraction cache, with a pointed error if it has not been built."""
    if not path.exists():
        raise FileNotFoundError(
            f"no extraction cache at {path}; run scripts/build_index.py first "
            f"(it writes the cache while indexing)"
        )
    return [Page(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus everything needed to cite it."""

    text: str
    source: str  # document title (PDF filename stem), shown in citations
    source_type: str  # "paper" | "textbook"
    page: int  # 1-indexed page the chunk was extracted from
    chunk_index: int  # position of this chunk within its page
    # A document's apparatus -- reference list, acknowledgments, contents page --
    # rather than its exposition. Indexed either way and filtered at query time, so
    # the decision stays reversible without re-embedding and both arms of the
    # comparison come from one index. Set by ``chunk_pages``; the default is the safe
    # direction, since a chunk built by hand is content until something says
    # otherwise. Only the boolean is stored: the *reason* is a pure function of the
    # text (``structure.classify``), so it can be recomputed from the index whenever
    # a flag needs to be argued with, and cannot go stale against the rules.
    structural: bool = False

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

# Email addresses, stripped for a different reason than the junk above: chunk
# text is quoted verbatim to students, and the corpus carries addresses that
# should not be shown to them. One textbook's title page holds a "Sold to
# <buyer>" purchase watermark naming the person who bought it; paper title pages
# carry author contact addresses. Neither ever helps answer a question about deep
# learning -- they are noise in the embedding and a disclosure in the citation.
#
# Two forms, because matching only the ordinary one leaves 13 addresses in this
# corpus: papers routinely print a LaTeX-style shared-domain author list,
# "{first,second,third}@lab.example.org" (sometimes space- or pipe-separated, and
# wrapped across lines -- which is why text is canonicalised before this runs).
# The brace form is tried first and length-capped so it cannot run away; the
# longest real match here is 81 characters.
#
# Redaction happens in ``load_pdf``, the one place raw PDF becomes text the rest
# of the system handles, so no downstream path can read an unredacted page.
# Measured after the fact: 68 addresses removed, 0 address-shaped strings left.
_EMAIL = re.compile(
    r"\{[^{}]{1,160}\}@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return ``(page_number, text)`` for each page that has extractable text.

    Page numbers are 1-indexed to match how a reader cites them -- note that this
    is pypdf's page index, not the number printed on the page; front matter makes
    those differ. Pages whose extracted text is empty (e.g. a full-page figure)
    are skipped rather than emitted as blank chunks.
    """
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _EXTRACTION_JUNK.sub(" ", page.extract_text() or "")
        # Canonicalise whitespace *before* redacting: PDF extraction wraps lines
        # mid-construct, and an author list broken across a line break would
        # otherwise slip past the pattern. Chunking collapses whitespace anyway,
        # so doing it here just makes page text deterministic. Collapsed again
        # afterwards because the substitution leaves a space of its own behind.
        text = " ".join(_EMAIL.sub(" ", " ".join(text.split())).split())
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


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    source: str,
    source_type: str,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk already-extracted pages.

    Split out from :func:`chunk_pdf` so a caller that has already extracted the
    pages -- the rebuild script, which caches the text for ground-truth lookup --
    can chunk them without a second extraction pass, and without reimplementing
    this loop. A copy of it would drift from the real one, and the index it built
    would silently stop matching what the pipeline produces.

    Chunking never crosses a page boundary, so every chunk cites exactly one
    page. The tradeoff: a passage split across a page break lands in two chunks
    with no overlap bridging them. Acceptable for Phase 1; revisit if recall
    suffers on concepts that straddle pages.

    Each chunk is also classified as exposition or apparatus here, because this is
    where chunks come into existence and a chunk that reaches the store unclassified
    would be indexed as content by default. Classification is per *chunk*, not per
    page, and the difference is not academic: page 14 of the Lottery Ticket paper is
    the acknowledgments page, and it also carries four chunks of appendix prose --
    one of which is the top hit for a real question about overfitting. Filtering that
    page would have deleted the answer to remove the apparatus.
    """
    count_tokens = count_tokens or token_counter()  # built once, reused across pages
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for idx, piece in enumerate(chunk_text(text, count_tokens, max_tokens, overlap_tokens)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    source_type=source_type,
                    page=page_number,
                    chunk_index=idx,
                    structural=is_structural(piece),
                )
            )
    return chunks


def chunk_cached_pages(
    pages: Iterable[Page],
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk the extraction cache, grouped back into documents.

    The cache is a flat page stream; :func:`chunk_pages` works a document at a time
    because ``source`` and ``source_type`` are per document. Shared by
    ``build_index.py --from-cache`` and ``audit_structure.py`` so the audit describes
    the chunks a rebuild would actually produce -- two copies of this regrouping
    would drift, and the audit would then be reporting on chunks nobody indexes.
    """
    by_document: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for page in pages:
        by_document.setdefault((page.source, page.source_type), []).append((page.page, page.text))
    count_tokens = count_tokens or token_counter()
    chunks: list[Chunk] = []
    for (source, source_type), page_list in sorted(by_document.items()):
        chunks += chunk_pages(
            sorted(page_list), source, source_type, count_tokens, max_tokens, overlap_tokens
        )
    return chunks


def chunk_pdf(
    path: Path,
    source_type: str,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Load a PDF and chunk it page by page.

    Papers and textbooks share these settings deliberately: the budget is set by
    the embedding model's input window, which does not care what kind of
    document the text came from.
    """
    return chunk_pages(
        load_pdf(path), path.stem, source_type, count_tokens, max_tokens, overlap_tokens
    )
