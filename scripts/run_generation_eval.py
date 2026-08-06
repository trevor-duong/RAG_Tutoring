"""Score generation against eval/questions.jsonl.

    python scripts/run_generation_eval.py [--save eval/generation-baseline.json]
                                          [--answers eval/generation-answers.jsonl]
                                          [--model claude-sonnet-5] [--k 5]

Needs ANTHROPIC_API_KEY. Makes one model call per question -- 37 by default, so this
costs real money and real time; there is no caching between runs on purpose, because a
cached answer would make a prompt change look like it did nothing.

``--save`` writes metadata only, never answer text: an answer may quote its passages,
and the same reasoning that keeps eval/baseline.json free of chunk text applies. Use
``--answers`` to write the prose somewhere gitignored for the manual read, which is
the part of this eval that actually measures quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.getLogger("transformers").setLevel(logging.CRITICAL)

from rag_tutoring.evaluate import load_questions, unindexed_labels  # noqa: E402
from rag_tutoring.generate import Generator  # noqa: E402
from rag_tutoring.generate_eval import format_report, run  # noqa: E402
from rag_tutoring.store import VectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", type=Path, help="write the report as JSON (metadata only, no prose)")
    ap.add_argument("--answers", type=Path, help="write answer text here for the manual read")
    ap.add_argument("--model", help="override the configured generation model")
    ap.add_argument("--k", type=int, default=5, help="passages handed to the model per question")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY is not set -- generation cannot be scored", file=sys.stderr)
        return 2

    questions = load_questions()
    store = VectorStore()
    if store.count() == 0:
        print("index is empty -- build it before evaluating", file=sys.stderr)
        return 2

    # Same guard as the retrieval eval: a label naming a page the index does not hold
    # is an unsatisfiable miss that reads as a quality problem rather than a bug.
    missing = unindexed_labels(store, questions)
    if missing:
        print(
            f"{len(missing)} labelled pages are not in the index; refusing to score",
            file=sys.stderr,
        )
        for label in missing[:10]:
            print(f"  {label.source} p{label.page}", file=sys.stderr)
        return 2

    import anthropic

    client = anthropic.Anthropic(api_key=key)
    generator = Generator(client, **({"model": args.model} if args.model else {}))

    print(f"generating {len(questions)} answers with {generator.model}...", file=sys.stderr)
    report = run(store, generator, questions, k=args.k)
    print(format_report(report))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report.to_json(), indent=2) + "\n")
        print(f"\nwrote {args.save} (metadata only)", file=sys.stderr)

    if args.answers:
        args.answers.parent.mkdir(parents=True, exist_ok=True)
        by_id = {o.question_id: o for o in report.outcomes}
        with args.answers.open("w") as f:
            for question in questions:
                text = report.answers.get(question.id)
                if text is None:
                    continue
                outcome = by_id[question.id]
                f.write(
                    json.dumps(
                        {
                            "id": question.id,
                            "partition": outcome.partition,
                            "question": question.question,
                            "answer": text,
                        }
                    )
                    + "\n"
                )
        print(f"wrote {args.answers} -- read the miss and negative entries", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
