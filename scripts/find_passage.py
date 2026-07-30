"""Find the page that answers a question, for labelling eval ground truth.

    python scripts/find_passage.py "layer normalization" --source "Layer Norm"

Deliberately **lexical** -- a regex over extracted text, not a vector search.
Ground truth located with the retriever would make the eval circular: it could
only ever confirm that retrieval finds what retrieval found. This is the tool
that keeps eval/questions.jsonl independent of the thing it measures.

Page numbers printed here are the indexer's (pypdf's page index), which is what
``Chunk.page`` stores and what a label in questions.jsonl must use. They are not
the numbers printed on the page -- in Dive into Deep Learning the two differ by
18 -- so labels come from here and never from a PDF viewer.
"""

from __future__ import annotations

import argparse
import re
import sys

from rag_tutoring.ingest import read_pages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern", help="regular expression, matched case-insensitively")
    ap.add_argument("--source", default="", help="only search documents whose title contains this")
    ap.add_argument("--chars", type=int, default=260, help="characters of context to show")
    ap.add_argument("--max", type=int, default=12, help="stop printing after this many pages")
    args = ap.parse_args()

    try:
        pattern = re.compile(args.pattern, re.I)
    except re.error as exc:
        print(f"bad regular expression: {exc}", file=sys.stderr)
        return 2

    matched = 0
    for page in read_pages():
        if args.source.lower() not in page.source.lower():
            continue
        found = pattern.search(page.text)
        if not found:
            continue
        matched += 1
        if matched > args.max:
            continue
        start = max(0, found.start() - args.chars // 4)
        print(f"\n{page.source}  p{page.page}  [{page.source_type}]")
        print("    " + page.text[start : start + args.chars])

    if matched > args.max:
        print(f"\n... {matched - args.max} more page(s) not shown (--max to raise)")
    print(f"\n{matched} page(s) matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
