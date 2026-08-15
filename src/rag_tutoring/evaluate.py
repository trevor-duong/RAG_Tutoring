"""Measure retrieval quality against a hand-labelled question set.

Three demo queries looking right is not a measurement. This module turns
``eval/questions.jsonl`` into numbers: recall@k and MRR over questions with
known-correct sources, and a separate score-separation report over questions the
corpus deliberately cannot answer.

Two choices here matter more than the metrics themselves.

**Labels are (document, page), not chunk ids.** A chunk id encodes the chunk
setting -- re-chunking the corpus renames every chunk, which would invalidate
every label. Page numbers survive re-chunking, so the same labelled set can
score a chunking change, which is the whole reason to have it.

**Negatives are scored separately, never folded into recall.** Mixing questions
with no correct answer into MRR drags the aggregate toward zero and makes it
uninterpretable. They answer a different question -- can the system tell it has
nothing? -- and get their own report.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from rag_tutoring.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL,
    repo_root,
)
from rag_tutoring.ingest import corpus_documents


def questions_path() -> Path:
    """The eval set. A function, not a constant: it lives in the checkout, and only
    the eval harness reads it -- see the module docstring in ``config``."""
    return repo_root() / "eval" / "questions.jsonl"


# How the question is phrased, which is the axis that most changes the score.
# A question reusing the source's own term is an easier retrieval problem than
# one describing a symptom, so an aggregate over a set that is mostly the former
# looks like rigour and measures very little. Reported broken out, never merged.
STYLES = ("verbatim-term", "paraphrase", "symptom")

# Ranks are reported at these cutoffs. k=5 is what the API will actually send to
# the model; k=1 and k=10 bracket it (is the best hit first, and how much does a
# wider window buy).
CUTOFFS = (1, 3, 5, 10)


@dataclass(frozen=True)
class Label:
    """A page that genuinely answers the question, as the indexer numbers pages.

    ``page`` is pypdf's page index, which is what ``Chunk.page`` stores -- not
    the number printed on the page. Front matter makes those differ by 18 in
    ``Dive into Deep Learning``, so labels are derived from ``load_pdf`` output
    and never read off a PDF viewer.
    """

    source: str
    page: int


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    topic: str
    style: str
    sources: tuple[Label, ...]  # empty => the corpus cannot answer this
    note: str  # why these pages answer it, paraphrased -- no source text

    @property
    def is_negative(self) -> bool:
        return not self.sources


@dataclass(frozen=True)
class QuestionResult:
    """One question's outcome. Deliberately carries no retrieved text.

    Chunk text is copyrighted source material (and one textbook's extraction
    carries the purchaser's email in its footer), so results stay citable
    metadata: where the hit came from and how confident it was.
    """

    question: Question
    rank: int | None  # 1-based rank of the first labelled hit, None if absent
    top_score: float
    top_source: str
    top_page: int
    # How many of the top 5 were a document's apparatus -- a reference list, an
    # acknowledgments section, a contents page -- rather than exposition. Always 0
    # once the filter is on; the number is what the unfiltered arm is for. Recall
    # cannot show this: a boilerplate chunk crowding out a fourth good passage moves
    # no rate at all, and on the negatives, where nothing is labelled, recall is not
    # even defined.
    structural_in_top5: int = 0

    @property
    def hit_at(self) -> int:
        return self.rank if self.rank is not None else 10**9


def load_questions(path: Path | None = None) -> list[Question]:
    """Parse and validate the question set.

    Validation is strict because every mistake here is silent: a bad style tag
    quietly drops a question from a breakdown, and a duplicate id quietly
    overwrites a result.
    """
    path = path if path is not None else questions_path()
    questions: list[Question] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        rec = json.loads(line)
        qid = rec["id"]
        if qid in seen:
            raise ValueError(f"{path.name}:{lineno}: duplicate question id {qid!r}")
        seen.add(qid)
        if rec["style"] not in STYLES:
            raise ValueError(f"{path.name}:{lineno}: style {rec['style']!r} not in {STYLES}")
        if not rec["question"].strip():
            raise ValueError(f"{path.name}:{lineno}: empty question")
        questions.append(
            Question(
                id=qid,
                question=rec["question"],
                topic=rec["topic"],
                style=rec["style"],
                sources=tuple(Label(s["source"], int(s["page"])) for s in rec["sources"]),
                note=rec["note"],
            )
        )
    return questions


def unindexed_labels(store, questions: Iterable[Question]) -> list[Label]:
    """Labels naming a page the index does not contain.

    Any such label is an automatic miss no retriever could ever satisfy, so it
    would show up as a quality problem rather than the labelling bug it is.
    Checked before scoring, one lookup per distinct document.
    """
    wanted: dict[str, set[int]] = {}
    for q in questions:
        for label in q.sources:
            wanted.setdefault(label.source, set()).add(label.page)
    missing: list[Label] = []
    for source, pages in sorted(wanted.items()):
        present = store.pages_present(source)
        missing += [Label(source, p) for p in sorted(pages - present)]
    return missing


def first_hit_rank(retrieved: Sequence[tuple[str, int]], labels: Iterable[Label]) -> int | None:
    """1-based rank of the first retrieved ``(source, page)`` that is labelled.

    ``None`` when no labelled page appears at all. Note what this makes recall
    mean: a retrieved passage that genuinely answers the question but was never
    labelled counts as a miss. The labels are not exhaustive -- 37 documents
    overlap heavily on these topics -- so every number here is a **lower bound**
    on true quality. That is the standard position with incomplete relevance
    judgements, and it is the honest one: the set is fixed, so the numbers stay
    comparable across chunking changes even if they understate absolute quality.
    """
    wanted = {(label.source, label.page) for label in labels}
    for rank, hit in enumerate(retrieved, start=1):
        if hit in wanted:
            return rank
    return None


def recall_at_k(results: Sequence[QuestionResult], k: int) -> float:
    """Fraction of questions with a labelled page in the top ``k``."""
    if not results:
        return 0.0
    return sum(r.hit_at <= k for r in results) / len(results)


def mrr(results: Sequence[QuestionResult]) -> float:
    """Mean reciprocal rank; a question with no hit contributes 0."""
    if not results:
        return 0.0
    return sum(1.0 / r.rank if r.rank else 0.0 for r in results) / len(results)


@dataclass(frozen=True)
class Report:
    provenance: dict
    labeled: tuple[QuestionResult, ...]
    negatives: tuple[QuestionResult, ...]

    def by_style(self) -> dict[str, tuple[QuestionResult, ...]]:
        return {
            style: tuple(r for r in self.labeled if r.question.style == style) for style in STYLES
        }

    def by_source_type(self, min_n: int = 3) -> dict[str, tuple[QuestionResult, ...]]:
        """Recall split by paper vs textbook.

        Groups smaller than ``min_n`` are dropped rather than printed: a rate
        over two questions is not a rate, and a row reading 1.00 invites being
        read as a result. Questions whose labels span both types land in their
        own group, which is usually one of the ones dropped.
        """
        types = {path.stem: kind for path, kind in corpus_documents()}
        grouped: dict[str, list[QuestionResult]] = {}
        for r in self.labeled:
            kinds = {types.get(label.source, "unknown") for label in r.question.sources}
            key = kinds.pop() if len(kinds) == 1 else "spans both"
            grouped.setdefault(key, []).append(r)
        return {k: tuple(v) for k, v in sorted(grouped.items()) if len(v) >= min_n}

    def controlled_pairs(self) -> list[tuple[QuestionResult, ...]]:
        """Questions sharing an identical label set but phrased differently.

        The per-style aggregate confounds phrasing with topic difficulty: maybe
        symptom questions score worse because the topics they happen to cover are
        harder. These pairs remove that -- same target pages, different wording,
        so the rank difference is attributable to the wording. Detected rather
        than declared: any two questions with the same labels form a pair.
        """
        groups: dict[frozenset, list[QuestionResult]] = {}
        for r in self.labeled:
            groups.setdefault(frozenset(r.question.sources), []).append(r)
        return [
            tuple(sorted(g, key=lambda r: STYLES.index(r.question.style)))
            for g in groups.values()
            if len(g) > 1 and len({r.question.style for r in g}) > 1
        ]

    def abstention_sweep(self, thresholds: Sequence[float], k: int = 5) -> list[tuple]:
        """What a "refuse to answer below score X" rule would cost and buy.

        Both sides, because either alone is misleading: how many good answers the
        rule would throw away, and how many unanswerable questions it would
        correctly refuse. No single threshold is recommended from a handful of
        negatives -- the point is whether the two score distributions separate
        at all.

        The cost side counts only questions that *did* retrieve a correct source
        within ``k``, not every labelled question. Refusing one retrieval already
        failed costs nothing -- a wrong answer is not what the threshold would be
        discarding. Note this is a narrower population than "answerable": all 32
        labelled questions are answerable by the corpus; these are the ones the
        retriever actually answered.
        """
        succeeded = [r for r in self.labeled if r.hit_at <= k]
        rows = []
        for t in thresholds:
            lost = sum(r.top_score < t for r in succeeded)
            refused = sum(r.top_score < t for r in self.negatives)
            rows.append((t, lost, len(succeeded), refused, len(self.negatives)))
        return rows


def evaluate(
    store,
    questions: Sequence[Question],
    k: int = max(CUTOFFS),
    include_structural: bool = False,
) -> Report:
    """Run every question through the store once and score the ranked results.

    One query per question at the largest cutoff, sliced for the smaller ones --
    re-querying per k would repeat identical work and could not disagree.

    ``include_structural`` selects the arm. The default matches what the API serves;
    passing ``True`` reproduces the pre-filter index from the same collection, which
    is what makes the two numbers comparable -- same chunks, same embeddings, one
    variable.
    """
    labeled: list[QuestionResult] = []
    negatives: list[QuestionResult] = []
    for q in questions:
        hits = store.query(q.question, k=k, include_structural=include_structural)
        result = QuestionResult(
            question=q,
            rank=first_hit_rank([(h.source, h.page) for h in hits], q.sources),
            top_score=hits[0].score if hits else 0.0,
            top_source=hits[0].source if hits else "",
            top_page=hits[0].page if hits else 0,
            structural_in_top5=sum(h.structural for h in hits[:5]),
        )
        (negatives if q.is_negative else labeled).append(result)
    provenance = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "collection": store.collection_name,
        "chunks_indexed": store.count(),
        # Both numbers, and the flag that says which one the scores describe. A
        # saved report that recorded only the total would look comparable to a
        # report from the other arm while measuring a different candidate set.
        "chunks_retrievable": store.count(include_structural=include_structural),
        "structural_filter": not include_structural,
        # From the index, not from data/raw: an interrupted ingestion would
        # otherwise report the corpus size while the scores describe less of it.
        "documents_indexed": len(store.sources()),
        "documents_on_disk": len(corpus_documents()),
        "k": k,
    }
    return Report(provenance, tuple(labeled), tuple(negatives))


def _line(name: str, results: Sequence[QuestionResult]) -> str:
    cells = "  ".join(f"{recall_at_k(results, k):5.2f}" for k in CUTOFFS)
    return f"  {name:22s} {len(results):3d}   {cells}   {mrr(results):5.3f}"


def format_report(report: Report) -> str:
    """Human-readable report. Provenance first: a score with no index state
    attached cannot answer "did my change help?"."""
    p = report.provenance
    out = [
        "RETRIEVAL EVAL",
        "",
        f"  model      {p['embedding_model']}",
        f"  chunking   {p['chunk_max_tokens']} word-pieces, {p['chunk_overlap_tokens']} overlap",
        f"  index      {p['collection']}  {p['chunks_indexed']:,} chunks from "
        f"{p['documents_indexed']} of {p['documents_on_disk']} documents",
        f"  filter     structural chunks {'EXCLUDED' if p['structural_filter'] else 'INCLUDED'}"
        f"  -- {p['chunks_retrievable']:,} of {p['chunks_indexed']:,} chunks searchable",
        "",
        f"  {'':22s}   n   " + "  ".join(f"r@{k:<3d}" for k in CUTOFFS) + "     MRR",
        _line("all labelled", report.labeled),
        "",
        "  by phrasing",
    ]
    out += [_line(f"  {style}", rs) for style, rs in report.by_style().items() if rs]
    # Say how many questions the min_n filter drops, not just that it drops some.
    # Otherwise the column does not sum to n and the reader cannot tell whether
    # that is a filter or a bug -- the rows look like a partition and are not.
    by_type = report.by_source_type()
    hidden = len(report.labeled) - sum(len(rs) for rs in by_type.values())
    out += ["", f"  by source type  ({hidden} in groups under 3 questions, not shown)"]
    out += [_line(f"  {kind}", rs) for kind, rs in by_type.items()]

    pairs = report.controlled_pairs()
    if pairs:
        out += [
            "",
            f"  same labels, different phrasing ({len(pairs)} pairs) -- rank of the first "
            "labelled hit",
        ]
        for group in pairs:
            cells = "   ".join(
                f"{r.question.style}: {r.rank if r.rank else 'miss':>4}" for r in group
            )
            out.append(f"    {group[0].question.topic:26s} {cells}")

    misses = [r for r in report.labeled if r.hit_at > max(CUTOFFS)]
    if misses:
        out += ["", f"  missed entirely ({len(misses)}) -- nothing labelled in top {max(CUTOFFS)}"]
        out += [
            f"    {r.question.id:24s} {r.question.style:14s} "
            f"top hit: {r.top_source[:38]:38s} p{r.top_page:<4d} {r.top_score:.3f}"
            for r in misses
        ]

    # Only ever non-empty on the unfiltered arm, and that is the point of printing
    # it: it is the size of the problem the filter exists to fix, stated in the unit
    # that matters -- slots on the page a student actually reads. Recall never sees
    # this, because a boilerplate chunk crowding out a fourth good passage changes no
    # rate, and on the negatives there is no rate to change.
    everything = (*report.labeled, *report.negatives)
    crowded = [r for r in everything if r.structural_in_top5]
    if crowded:
        taken = sum(r.structural_in_top5 for r in crowded)
        out += [
            "",
            "  APPARATUS IN THE TOP 5 -- reference lists, acknowledgments, contents pages",
            f"    {taken} of {5 * len(everything)} slots, on {len(crowded)} of "
            f"{len(everything)} questions",
        ]
        out += [
            f"    {r.question.id:26s} {r.structural_in_top5}/5"
            for r in sorted(crowded, key=lambda r: (-r.structural_in_top5, r.question.id))[:8]
        ]

    if report.negatives:
        # "retrieval succeeded", not "answerable": every labelled question is
        # answerable by the corpus. These are the ones the retriever answered, and
        # they are what a score threshold would actually be throwing away.
        hit_scores = [r.top_score for r in report.labeled if r.hit_at <= 5]
        neg_scores = [r.top_score for r in report.negatives]
        rows = [
            (f"retrieval succeeded (n={len(hit_scores)})", hit_scores),
            (f"unanswerable (n={len(neg_scores)})", neg_scores),
        ]
        width = max(len(label) for label, _ in rows)  # computed, not hand-counted
        out += ["", "  ABSTENTION -- questions the corpus cannot answer"]
        for label, scores in rows:
            # A group can be empty: recall@5 = 0 empties the first one. Finding
            # that out as min([]) blowing up after a 13-minute rebuild, with no
            # report at all, is the wrong way to learn a change made things worse.
            stats = (
                f"min {min(scores):.3f}  median {median(scores):.3f}  max {max(scores):.3f}"
                if scores
                else "no questions in this group"
            )
            out.append(f"    top-1 score, {label:{width}s}  {stats}")
        out += ["", "    threshold   good answers refused   unanswerable refused"]
        for t, lost, n_ans, refused, n_neg in report.abstention_sweep(
            (0.30, 0.35, 0.40, 0.45, 0.50)
        ):
            out.append(
                f"      {t:.2f}          {lost:2d} / {n_ans:<3d}            {refused:2d} / {n_neg}"
            )
    return "\n".join(out)


def to_json(report: Report) -> str:
    """Serialise for committing as a baseline. Metadata and scores only."""
    return json.dumps(
        {
            "provenance": report.provenance,
            "summary": {
                "n_labeled": len(report.labeled),
                "n_negatives": len(report.negatives),
                "structural_top5_slots": sum(
                    r.structural_in_top5 for r in (*report.labeled, *report.negatives)
                ),
                "mrr": round(mrr(report.labeled), 4),
                **{f"recall@{k}": round(recall_at_k(report.labeled, k), 4) for k in CUTOFFS},
                "by_style": {
                    style: {
                        "n": len(rs),
                        "mrr": round(mrr(rs), 4),
                        **{f"recall@{k}": round(recall_at_k(rs, k), 4) for k in CUTOFFS},
                    }
                    for style, rs in report.by_style().items()
                    if rs
                },
            },
            "questions": [
                {
                    "id": r.question.id,
                    "style": r.question.style,
                    "topic": r.question.topic,
                    "rank": r.rank,
                    "top_score": round(r.top_score, 4),
                    "top_source": r.top_source,
                    "top_page": r.top_page,
                    "structural_in_top5": r.structural_in_top5,
                }
                for r in (*report.labeled, *report.negatives)
            ],
        },
        indent=2,
    )
