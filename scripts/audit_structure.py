"""Audit what the structural filter would remove, before it removes it.

    python scripts/audit_structure.py [--sample reference-list] [--n 8]

``structure.classify`` decides which chunks are a document's apparatus -- reference
lists, acknowledgments, front matter -- rather than its exposition. A false positive
there deletes a real explanation from the index, and nothing downstream would ever
report it: the retrieval eval only checks 32 labelled pages, so a filter eating
content on any of the other 2,600 is invisible to it.

This is the check that can see it. Two findings do the work:

* **Flagged fraction per document.** A rule misfiring on one document's formatting
  shows up as that document sitting far above the rest. Papers land in the 5-15%
  band (their bibliographies are a real fraction of their pages); a textbook should
  be low single digits outside its front and back matter.
* **Labelled pages that lose every chunk.** A page the eval set says answers a
  question, filtered out entirely, is a retrieval failure the eval would score as
  poor quality rather than as the bug it is. Exits non-zero on any.

Reads the extraction cache and re-chunks it with the same call ``build_index.py``
makes, so it audits the chunks that would actually be indexed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from rag_tutoring.evaluate import load_questions
from rag_tutoring.ingest import chunk_cached_pages, read_pages
from rag_tutoring.structure import classify


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", help="print flagged chunks with this reason, to eyeball them")
    ap.add_argument("--n", type=int, default=8, help="how many to print with --sample")
    args = ap.parse_args()

    chunks = chunk_cached_pages(read_pages())
    reasons = [classify(c.text) for c in chunks]
    flagged = [(c, r) for c, r in zip(chunks, reasons, strict=True) if r]

    if args.sample:
        picked = [(c, r) for c, r in flagged if r == args.sample]
        print(f"{len(picked)} chunk(s) flagged as {args.sample}; showing {args.n} spread evenly\n")
        for c, _ in picked[:: max(len(picked) // args.n, 1)][: args.n]:
            print(f"--- {c.id}\n    {c.text[:300]}\n")
        return 0

    print(
        f"A. chunks / flagged:  {len(chunks):,} / {len(flagged):,} "
        f"({len(flagged) / len(chunks):.1%})"
    )
    print("B. by reason:")
    for reason, n in Counter(r for _, r in flagged).most_common():
        print(f"     {reason:20s} {n:5,d}")

    per_doc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c, r in zip(chunks, reasons, strict=True):
        per_doc[c.source][0] += 1
        per_doc[c.source][1] += bool(r)
    ranked = sorted(per_doc.items(), key=lambda kv: kv[1][1] / kv[1][0], reverse=True)
    print("\nC. flagged fraction per document, worst first:")
    for source, (total, hit) in ranked:
        bar = "#" * round(40 * hit / total)
        print(f"     {hit / total:5.1%} {hit:4d}/{total:<5d} {source[:44]:44s} {bar}")

    # A labelled page with no surviving chunk can never be retrieved, so the eval
    # would score it as a miss and the number would read as poor retrieval.
    kept_pages: dict[str, set[int]] = defaultdict(set)
    for c, r in zip(chunks, reasons, strict=True):
        if not r:
            kept_pages[c.source].add(c.page)
    wiped = [
        label
        for q in load_questions()
        for label in q.sources
        if label.page not in kept_pages[label.source]
    ]
    print(f"\nD. labelled pages losing every chunk: {len(wiped)}")
    for label in wiped:
        print(f"     {label.source} p{label.page}")

    warnings = [
        f"{source} is {hit / total:.0%} flagged -- check the rule against its formatting"
        for source, (total, hit) in ranked
        if hit / total > 0.30
    ]
    print()
    for w in warnings:
        print("WARN:", w)
    if wiped:
        print(f"FAIL: {len(wiped)} labelled page(s) would become unretrievable")
        return 1
    print("OK -- no labelled page loses all of its chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
