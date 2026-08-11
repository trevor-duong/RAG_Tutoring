"""Rebuild the vector index over the whole corpus, from scratch.

    python scripts/build_index.py [--from-cache]

A full reset and rebuild rather than adding documents incrementally. Same cost
either way -- embeddings are local -- and it makes the index's provenance
unambiguous: every chunk in it came from the current chunk setting, so the
numbers in eval/baseline.json describe something reproducible. Incremental adds
leave chunks from older settings behind (see ``VectorStore.reset``).

Extraction is the slow part (~14 minutes for this corpus, most of it one 709 MB
illustrated textbook), so each page is extracted exactly once and the text is
cached to ``config.PAGES_CACHE`` on the way through. ``audit_corpus.py`` and
``find_passage.py`` read that cache instead of paying for extraction again.

``--from-cache`` skips extraction and re-chunks that cache instead, for a change
that alters chunking or metadata but not extraction -- 15 seconds rather than 14
minutes. Valid **only** while ``load_pdf`` is unchanged, since the cache is its
output; the script cannot verify that, so it checks what it can (same documents,
same page count) and the real guard is downstream: re-running ``run_eval.py`` on an
unchanged setting has to reproduce the saved baseline exactly, and a stale cache
would not.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.getLogger("pypdf").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

from rag_tutoring.config import PAGES_CACHE  # noqa: E402
from rag_tutoring.ingest import (  # noqa: E402
    Page,
    chunk_cached_pages,
    chunk_pages,
    corpus_documents,
    load_pdf,
    read_pages,
    token_counter,
    write_pages,
)
from rag_tutoring.store import VectorStore  # noqa: E402


def rebuild_from_cache(store: VectorStore, count_tokens) -> int:
    """Re-chunk and re-index the extraction cache. Returns a process exit code."""
    pages = read_pages()
    cached = {p.source for p in pages}
    on_disk = {path.stem for path, _ in corpus_documents()}
    if cached != on_disk:
        print(
            f"cache holds {len(cached)} documents but {len(on_disk)} are on disk "
            f"(missing: {sorted(on_disk - cached)[:3]}); run a full build",
            file=sys.stderr,
        )
        return 2

    started = time.time()
    chunks = chunk_cached_pages(pages, count_tokens)
    store.add(chunks)
    structural = sum(c.structural for c in chunks)
    print(
        f"indexed {store.count():,} chunks from {len(pages):,} cached pages in "
        f"{time.time() - started:.0f}s; {structural:,} flagged structural "
        f"({structural / len(chunks):.1%}), searchable by default: "
        f"{len(chunks) - structural:,}"
    )
    print("OK -- now run scripts/audit_structure.py, then scripts/run_eval.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="re-chunk the extraction cache instead of re-reading the PDFs",
    )
    args = ap.parse_args()

    docs = corpus_documents()
    if not docs:
        print("no PDFs found under data/raw/{papers,textbooks}/", file=sys.stderr)
        return 2

    count_tokens = token_counter()  # built once; the vocabulary is shared by every document
    store = VectorStore()
    print(f"{store.collection_name}: {store.count()} chunks indexed, resetting", flush=True)
    store.reset()

    if args.from_cache:
        return rebuild_from_cache(store, count_tokens)

    all_pages: list[Page] = []
    total_chunks = 0
    started = time.time()
    for n, (path, source_type) in enumerate(docs, start=1):
        began = time.time()
        pages = load_pdf(path)
        all_pages += [Page(path.stem, source_type, number, text) for number, text in pages]
        # Same code path the pipeline uses -- chunk_pdf is chunk_pages over
        # load_pdf -- so an index built here cannot drift from one built by it.
        chunks = chunk_pages(pages, path.stem, source_type, count_tokens)
        store.add(chunks)
        total_chunks += len(chunks)
        print(
            f"  [{n:2d}/{len(docs)}] {path.stem[:48]:48s} "
            f"{len(pages):4d}p {len(chunks):5d}c  {time.time() - began:6.1f}s",
            flush=True,
        )

    cached = write_pages(PAGES_CACHE, all_pages)
    print(
        f"\nindexed {store.count():,} chunks from {cached:,} pages "
        f"in {time.time() - started:.0f}s; cached page text to {PAGES_CACHE}"
    )
    if store.count() != total_chunks:
        print(
            f"MISMATCH: built {total_chunks} chunks but the collection holds "
            f"{store.count()} -- ids may be colliding",
            file=sys.stderr,
        )
        return 1
    print("OK -- now run scripts/audit_corpus.py, then scripts/run_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
