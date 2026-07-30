"""Tests for the eval metrics.

The metrics take ranked ``(source, page)`` tuples rather than a store, so these
run without an index or an embedding model. That separation is the point: the
arithmetic is what silently produces a plausible-but-wrong number, so it is
worth pinning independently of retrieval.
"""

from __future__ import annotations

import pytest

from rag_tutoring.evaluate import (
    Label,
    Question,
    QuestionResult,
    first_hit_rank,
    load_questions,
    mrr,
    recall_at_k,
)


def question(qid: str, style: str = "verbatim-term", pages: tuple[int, ...] = (1,)) -> Question:
    return Question(
        id=qid,
        question="q?",
        topic="t",
        style=style,
        sources=tuple(Label("Doc", p) for p in pages),
        note="",
    )


def result(rank: int | None, style: str = "verbatim-term", score: float = 0.5) -> QuestionResult:
    return QuestionResult(
        question=question("q", style),
        rank=rank,
        top_score=score,
        top_source="Doc",
        top_page=1,
    )


def test_first_hit_rank_is_one_based():
    retrieved = [("Other", 3), ("Doc", 7), ("Doc", 2)]
    assert first_hit_rank(retrieved, [Label("Doc", 7)]) == 2


def test_first_hit_rank_returns_the_earliest_of_several_labels():
    retrieved = [("Doc", 9), ("Doc", 7)]
    assert first_hit_rank(retrieved, [Label("Doc", 7), Label("Doc", 9)]) == 1


def test_first_hit_rank_is_none_when_nothing_labelled_is_retrieved():
    assert first_hit_rank([("Doc", 1)], [Label("Doc", 2)]) is None


def test_page_must_match_not_just_the_document():
    """The right document at the wrong page is a miss.

    A textbook chapter spans dozens of pages; crediting the document alone would
    let a hit anywhere in Dive into Deep Learning count for every question, and
    the number would stop meaning anything.
    """
    assert first_hit_rank([("Doc", 400)], [Label("Doc", 42)]) is None


def test_recall_counts_a_question_once_however_many_labels_it_hits():
    results = [result(1), result(2), result(None)]
    assert recall_at_k(results, 5) == pytest.approx(2 / 3)


def test_recall_at_k_respects_the_cutoff():
    results = [result(1), result(4)]
    assert recall_at_k(results, 3) == pytest.approx(0.5)
    assert recall_at_k(results, 4) == pytest.approx(1.0)


def test_mrr_weights_by_rank_and_scores_misses_as_zero():
    # ranks 1, 4, miss -> (1 + 0.25 + 0) / 3
    assert mrr([result(1), result(4), result(None)]) == pytest.approx(1.25 / 3)


def test_empty_sets_do_not_divide_by_zero():
    assert recall_at_k([], 5) == 0.0
    assert mrr([]) == 0.0


def test_controlled_pairs_group_questions_with_identical_labels():
    from rag_tutoring.evaluate import Report

    def res(qid, style, pages):
        return QuestionResult(question(qid, style, pages), 1, 0.5, "Doc", 1)

    report = Report(
        provenance={},
        labeled=(
            res("named", "verbatim-term", (5, 6)),
            res("symptom", "symptom", (6, 5)),  # same labels, order should not matter
            res("elsewhere", "verbatim-term", (9,)),
        ),
        negatives=(),
    )
    pairs = report.controlled_pairs()
    assert len(pairs) == 1, "only the two questions sharing a label set form a pair"
    assert {r.question.id for r in pairs[0]} == {"named", "symptom"}


def test_controlled_pairs_ignores_same_style_duplicates():
    """Two identically-phrased questions on the same pages compare nothing."""
    from rag_tutoring.evaluate import Report

    def res(qid):
        return QuestionResult(question(qid, "verbatim-term", (5,)), 1, 0.5, "Doc", 1)

    report = Report(provenance={}, labeled=(res("a"), res("b")), negatives=())
    assert report.controlled_pairs() == []


def test_a_question_with_no_labelled_sources_is_a_negative():
    assert question("q", pages=()).is_negative
    assert not question("q", pages=(1,)).is_negative


def test_the_committed_question_set_is_valid():
    """Guards the eval set itself: a duplicate id would silently overwrite a
    result, and an unknown style tag would silently drop a question from the
    per-phrasing breakdown."""
    questions = load_questions()
    labeled = [q for q in questions if not q.is_negative]
    assert len(labeled) >= 20, "the roadmap calls for a 20-30 question set"
    assert any(q.is_negative for q in questions), "needs unanswerable questions too"
    assert all(q.note for q in questions), "every label needs a stated justification"
    # Pages are the indexer's numbering, which is 1-based.
    assert all(label.page >= 1 for q in questions for label in q.sources)


def test_question_ids_are_unique(tmp_path):
    path = tmp_path / "dupes.jsonl"
    row = (
        '{"id": "a", "topic": "t", "style": "symptom", "question": "q", "sources": [], "note": "n"}'
    )
    path.write_text(f"{row}\n{row}\n")
    with pytest.raises(ValueError, match="duplicate question id"):
        load_questions(path)


def test_unknown_style_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id": "a", "topic": "t", "style": "tricky", "question": "q",'
        ' "sources": [], "note": "n"}\n'
    )
    with pytest.raises(ValueError, match="style"):
        load_questions(path)


PROVENANCE = {
    "embedding_model": "m",
    "chunk_max_tokens": 240,
    "chunk_overlap_tokens": 60,
    "collection": "c",
    "chunks_indexed": 1,
    "documents_indexed": 1,
    "documents_on_disk": 1,
}


def test_report_states_how_many_questions_the_source_type_filter_hides():
    """The by-source-type rows are not a partition and must not read as one.

    Groups under three questions are dropped, so the column sums to less than n
    -- a reader who cannot tell a filter from a bug has been handed something
    worse than no number. Here both questions fall in one undersized group, so
    every row is dropped and the header has to account for all of them.
    """
    from rag_tutoring.evaluate import Report, format_report

    def res(qid):
        return QuestionResult(question(qid, "verbatim-term", (5,)), 1, 0.5, "Doc", 1)

    report = Report(provenance=PROVENANCE, labeled=(res("a"), res("b")), negatives=())
    line = next(ln for ln in format_report(report).splitlines() if "by source type" in ln)
    assert "2 in groups under 3" in line, line


def test_abstention_block_survives_zero_recall():
    """A run where nothing is retrieved must still produce a report.

    ``recall@5 = 0`` empties the succeeded group, and ``min([])`` raises -- losing
    the entire report of a 13-minute rebuild at exactly the moment it is saying
    a change made things much worse.
    """
    from rag_tutoring.evaluate import Report, format_report

    missed = QuestionResult(question("a", "symptom", (5,)), None, 0.4, "Doc", 1)
    negative = QuestionResult(question("n", "symptom", ()), None, 0.3, "Doc", 1)
    text = format_report(Report(provenance=PROVENANCE, labeled=(missed,), negatives=(negative,)))
    assert "no questions in this group" in text
