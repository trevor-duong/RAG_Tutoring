"""Audit the whole corpus for failures that do not raise.

    python scripts/audit_corpus.py

The regression net for a specific class of bug this project has been bitten by
more than once: something that corrupts or drops data while every test stays
green and retrieval still looks plausible. Chunks silently truncated past the
embedding window, a lone surrogate that kills the tokenizer, mis-decoded font
glyphs landing in control code points, an email address surviving into text that
gets quoted to a student. None of those raise on their own.

Reads the extraction cache written by ``build_index.py``, so it audits exactly
the text that was indexed rather than re-extracting and hoping the two agree.
Exits non-zero on any failure, so it can gate a rebuild.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from rag_tutoring.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    MODEL_MAX_TOKENS,
)
from rag_tutoring.ingest import (
    Chunk,
    corpus_documents,
    pack_words,
    read_pages,
    token_counter,
)

# Anything that still looks like an address after redaction -- deliberately
# looser than the redaction pattern itself, so a form that pattern misses is
# caught here rather than shipped.
ADDRESS_SHAPED = re.compile(r"\S*@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def main() -> int:
    pages = read_pages()
    count = token_counter()
    failures: list[str] = []
    warnings: list[str] = []

    # Source stems become chunk id prefixes, so a duplicate stem collides ids
    # across two different documents.
    stems = Counter(path.stem for path, _ in corpus_documents())
    if dupes := sorted(s for s, n in stems.items() if n > 1):
        failures.append(f"duplicate source stems, which would collide chunk ids: {dupes}")

    ids: dict[str, int] = defaultdict(int)
    tally = dict(chunks=0, words=0, invisible=0, oversized=0, empty=0, short=0, control=0)
    tally.update(emails=0, tokenizer_errors=0)
    control_chars: Counter[str] = Counter()
    max_index = max_page = 0
    documents = set()

    for page in pages:
        documents.add(page.source)
        words = page.text.split()
        if not words:
            continue
        costs = [count(w) for w in words]
        spans = pack_words(costs, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)
        covered: set[int] = set()

        for index, (start, end) in enumerate(spans):
            covered.update(range(start, end))
            text = " ".join(words[start:end])
            ids[Chunk(text, page.source, page.source_type, page.page, index).id] += 1

            if sum(costs[start:end]) + 2 > MODEL_MAX_TOKENS:  # +2 = [CLS]/[SEP]
                tally["oversized"] += 1
            if not text.strip():
                tally["empty"] += 1
            elif len(words[start:end]) < 5:
                tally["short"] += 1
            bad = [c for c in text if unicodedata.category(c) in ("Cc", "Cs")]
            if bad:
                tally["control"] += 1
                control_chars.update(bad)
            # Checked with ADDRESS_SHAPED, not with ingest._EMAIL: the redaction
            # pattern was already applied to this text, so re-running it here
            # could only ever find nothing. A check that cannot fail is worse
            # than no check, because it reads as coverage.
            if ADDRESS_SHAPED.search(text):
                tally["emails"] += 1
            try:
                count(text)
            except Exception:  # noqa: BLE001 -- any tokenizer failure at all is the finding
                tally["tokenizer_errors"] += 1
            max_index = max(max_index, index)

        tally["chunks"] += len(spans)
        tally["words"] += len(words)
        tally["invisible"] += len(words) - len(covered)  # words no chunk covers
        max_page = max(max_page, page.page)

    collisions = {cid: n for cid, n in ids.items() if n > 1}
    print(
        f"A. documents / pages / chunks:            {len(documents)} / {len(pages)} / "
        f"{tally['chunks']:,}"
    )
    print(f"B. words / reaching no embedding:         {tally['words']:,} / {tally['invisible']}")
    print(f"C. chunks over {MODEL_MAX_TOKENS} word-pieces:           {tally['oversized']}")
    print(f"D. empty / very short (<5 words):         {tally['empty']} / {tally['short']}")
    print(
        f"E. chunks with control or surrogate chars:{tally['control']:4d}  "
        f"{dict(control_chars.most_common(5))}"
    )
    print(f"F. chunks containing an email address:    {tally['emails']}")
    print(f"G. tokenizer failures:                    {tally['tokenizer_errors']}")
    print(f"H. unique chunk ids / collisions:         {len(ids):,} / {len(collisions)}")
    print(
        f"I. max chunk_index / max page:            {max_index} / {max_page}"
        f"  (ids pad to 2 and 4 digits)"
    )

    for bad, message in [
        (tally["invisible"], f"{tally['invisible']} words reach no embedding"),
        (tally["oversized"], f"{tally['oversized']} chunks would be silently truncated"),
        (tally["empty"], f"{tally['empty']} empty chunks"),
        (tally["control"], f"{tally['control']} chunks carry control/surrogate characters"),
        (tally["emails"], f"{tally['emails']} chunks still contain an email address"),
        (tally["tokenizer_errors"], f"{tally['tokenizer_errors']} chunks break the tokenizer"),
        (len(collisions), f"{len(collisions)} chunk-id collisions"),
    ]:
        if bad:
            failures.append(message)
    if max_index > 99:
        warnings.append(f"chunk_index {max_index} exceeds the 2-digit zero-padding in chunk ids")
    if max_page > 9999:
        warnings.append(f"page {max_page} exceeds the 4-digit zero-padding in chunk ids")
    if missing := sorted({path.stem for path, _ in corpus_documents()} - documents):
        warnings.append(f"{len(missing)} document(s) produced no text: {missing[:3]}")

    print()
    for w in warnings:
        print("WARN:", w)
    for f in failures:
        print("FAIL:", f)
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
