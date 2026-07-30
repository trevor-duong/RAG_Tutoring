"""Score retrieval against eval/questions.jsonl.

    python scripts/run_eval.py [--save eval/baseline.json] [--k 10]

Refuses to score if any label names a page the index does not contain: that
would be an automatic miss no retriever could satisfy, and it would read as a
quality problem rather than the labelling bug it is.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.CRITICAL)

from rag_tutoring.evaluate import (  # noqa: E402
    CUTOFFS,
    evaluate,
    format_report,
    load_questions,
    to_json,
    unindexed_labels,
)
from rag_tutoring.store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", type=Path, help="write the report as JSON (scores only, no text)")
    ap.add_argument("--k", type=int, default=max(CUTOFFS), help="retrieve this many per question")
    args = ap.parse_args()

    questions = load_questions()
    store = VectorStore()
    if store.count() == 0:
        print("index is empty -- ingest the corpus before evaluating", file=sys.stderr)
        return 2

    missing = unindexed_labels(store, questions)
    if missing:
        print(
            f"{len(missing)} label(s) name a page that is not indexed. Either the "
            f"label is wrong or the document was not ingested:",
            file=sys.stderr,
        )
        for label in missing:
            print(f"  {label.source} p{label.page}", file=sys.stderr)
        return 1

    report = evaluate(store, questions, k=args.k)
    print(format_report(report))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(to_json(report) + "\n")
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
