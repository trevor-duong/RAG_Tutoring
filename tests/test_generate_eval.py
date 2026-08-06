"""Tests for the generation eval's scoring logic.

No index, no key, no network: a fake store returns fixed hits and a fake generator
returns fixed answers, so what is under test is the partitioning and the counting.

That is the part worth testing. A miscounted partition would not raise -- it would
produce a report whose rows look plausible and describe the wrong questions, which is
the same failure mode as every measurement bug this project has logged.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_tutoring.evaluate import Label, Question
from rag_tutoring.generate import Answer, UngroundedCitation
from rag_tutoring.generate_eval import run, summarise


@dataclass(frozen=True)
class FakeRetrieved:
    text: str
    source: str
    source_type: str
    page: int
    score: float


class FakeStore:
    """Returns the given ``(source, page)`` pairs as hits, in order."""

    collection_name = "test-collection"

    def __init__(self, hits: list[tuple[str, int]]) -> None:
        self.hits = hits

    def query(self, text: str, k: int = 5):
        return [
            FakeRetrieved(
                text=f"passage about {source}",
                source=source,
                source_type="paper",
                page=page,
                score=0.9 - i / 100,
            )
            for i, (source, page) in enumerate(self.hits[:k])
        ]

    def count(self) -> int:
        return 9169

    def sources(self) -> list[str]:
        return ["Dropout", "GloVe"]


class FakeGenerator:
    """Returns a canned answer, or raises, for every question."""

    model = "fake-model"

    def __init__(self, text: str = "grounded [source 1]", cited=(1,), raises=None) -> None:
        self.text = text
        self.cited = cited
        self.raises = raises

    def answer(self, question: str, citations):
        if self.raises is not None:
            raise self.raises
        return Answer(text=self.text, cited=self.cited, model=self.model, prompt_version="1")


def question(qid: str, labels: list[tuple[str, int]], style: str = "verbatim-term") -> Question:
    return Question(
        id=qid,
        question=f"question {qid}?",
        topic="testing",
        style=style,
        sources=tuple(Label(source=s, page=p) for s, p in labels),
        note="",
    )


def test_a_retrieval_hit_is_partitioned_as_a_hit():
    store = FakeStore([("Dropout", 17), ("GloVe", 3)])
    report = run(store, FakeGenerator(), [question("q1", [("Dropout", 17)])])
    (outcome,) = report.outcomes
    assert outcome.partition == "hit"
    assert outcome.retrieval_rank == 1


def test_a_retrieval_miss_is_partitioned_as_a_miss():
    """The 16-question partition that carries the risk. If a miss were scored as a hit,
    the report would claim generation behaves well on exactly the cases it was built
    to interrogate."""
    store = FakeStore([("GloVe", 3), ("GloVe", 4)])
    report = run(store, FakeGenerator(), [question("q1", [("Dropout", 17)])])
    (outcome,) = report.outcomes
    assert outcome.partition == "miss"
    assert outcome.retrieval_rank is None


def test_an_unanswerable_question_is_a_negative_even_when_something_is_retrieved():
    """Retrieval always returns k passages, so a negative question still gets hits.

    Classifying by "did retrieval fail?" would file every negative under miss and
    collapse the two partitions the eval exists to keep apart.
    """
    store = FakeStore([("GloVe", 3)])
    report = run(store, FakeGenerator(), [question("q1", [])])
    (outcome,) = report.outcomes
    assert outcome.partition == "negative"


def test_citing_a_labelled_page_is_detected_through_the_passage_number():
    """``cited`` is 1-based into the retrieved list, so this crosses two indexes.

    An off-by-one here would attribute the citation to the neighbouring passage and
    silently invert the metric on adjacent-page questions.
    """
    store = FakeStore([("GloVe", 3), ("Dropout", 17)])
    labelled = question("q1", [("Dropout", 17)])
    assert run(store, FakeGenerator(cited=(2,)), [labelled]).outcomes[0].cited_a_labelled_page
    assert not run(store, FakeGenerator(cited=(1,)), [labelled]).outcomes[0].cited_a_labelled_page


def test_an_uncited_answer_does_not_count_as_citing_a_labelled_page():
    store = FakeStore([("Dropout", 17)])
    report = run(
        store, FakeGenerator(text="no markers here", cited=()), [question("q1", [("Dropout", 17)])]
    )
    outcome = report.outcomes[0]
    assert outcome.answered
    assert outcome.n_cited == 0
    assert not outcome.cited_a_labelled_page


def test_an_ungrounded_citation_is_counted_not_raised():
    """The rate is one of the things this eval exists to quantify, so a single bad
    answer must not abort a 37-question run partway through."""
    store = FakeStore([("Dropout", 17)])
    report = run(
        store,
        FakeGenerator(raises=UngroundedCitation("cites [source 9] of 5")),
        [question("q1", [("Dropout", 17)])],
    )
    (outcome,) = report.outcomes
    assert outcome.ungrounded
    assert not outcome.answered
    assert outcome.answer_chars == 0


def test_answer_text_is_kept_out_of_the_committed_artifact():
    """Metrics are committed; prose is not, because an answer may quote its passages."""
    store = FakeStore([("Dropout", 17)])
    report = run(
        store,
        FakeGenerator(text="quoted passage text [source 1]"),
        [question("q1", [("Dropout", 17)])],
    )
    assert report.answers["q1"] == "quoted passage text [source 1]"
    assert "quoted passage text" not in str(report.to_json())


def test_the_hedge_proxy_is_a_substring_check_and_is_labelled_as_one():
    store = FakeStore([("Dropout", 17)])
    q = [question("q1", [("Dropout", 17)])]
    hedged = FakeGenerator(text="These passages do not address this directly.", cited=())
    assert run(store, hedged, q).outcomes[0].hedge_phrase_present
    assert (
        not run(store, FakeGenerator(text="Dropout prevents co-adaptation."), q)
        .outcomes[0]
        .hedge_phrase_present
    )


def test_summarise_reports_n_even_when_the_partition_is_empty():
    """A rate with no denominator invites being read as a result -- and the negatives
    partition is n=5."""
    assert summarise([]) == {"n": 0}
