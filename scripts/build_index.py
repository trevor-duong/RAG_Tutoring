"""Rebuild the vector index over the whole corpus, from scratch.

    python scripts/build_index.py

A full reset and rebuild rather than adding documents incrementally. Same cost
either way -- embeddings are local -- and it makes the index's provenance
unambiguous: every chunk in it came from the current chunk setting, so the
numbers in eval/baseline.json describe something reproducible. Incremental adds
leave chunks from older settings behind (see ``VectorStore.reset``).

Extraction is the slow part (~14 minutes for this corpus, most of it one 709 MB
illustrated textbook), so each page is extracted exactly once and the text is
cached to ``config.PAGES_CACHE`` on the way through. ``audit_corpus.py`` and
``find_passage.py`` read that cache instead of paying for extraction again.
"""

from __future__ import annotations

import logging
import sys
import time

logging.getLogger("pypdf").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

from rag_tutoring.config import PAGES_CACHE  # noqa: E402
from rag_tutoring.ingest import (  # noqa: E402
    Page,
    chunk_pages,
    corpus_documents,
    load_pdf,
    token_counter,
    write_pages,
)
from rag_tutoring.store import VectorStore  # noqa: E402


def main() -> int:
    docs = corpus_documents()
    if not docs:
        print("no PDFs found under data/raw/{papers,textbooks}/", file=sys.stderr)
        return 2

    count_tokens = token_counter()  # built once; the vocabulary is shared by every document
    store = VectorStore()
    print(f"{store.collection_name}: {store.count()} chunks indexed, resetting", flush=True)
    store.reset()

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
