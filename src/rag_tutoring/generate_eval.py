"""Score generation on the two partitions that carry the risk.

Retrieval's eval asks "is a correct page in the top k?". This asks a different
question, on the cases where the answer to that one was *no*:

* **The misses.** recall@5 = 0.50 over 32 labelled questions, so ~16 questions hand
  the model five passages that do not contain a correct page. What it does there is
  the entire risk documented in ``eval/README.md`` -- and the partition needs no new
  labelling, because the existing labels already define it.
* **The negatives.** 5 questions the corpus cannot answer at all. The 07-30 finding
  was that a *similarity threshold* cannot separate these from answerable ones (the
  score ranges overlap). A model reading five passages and judging that they do not
  address the question is a different mechanism, not a scalar cutoff, so it can
  succeed where the threshold provably could not.

**What is measured here is mechanical, and it is not answer quality.** Whether an
answer is *faithful* needs a judge or a human; whether it cites, what it cites, and
whether every citation resolves are all checkable in code, and those are what this
module reports. The 21 misses and negatives are dumped for reading, because at that
size reading them is more honest than a proxy.

Results go to ``eval/generation-baseline.json``. ``eval/baseline.json`` is left
untouched: it is the frozen retrieval baseline, and a generation change must not be
able to move it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_tutoring.citations import cite
from rag_tutoring.config import EMBEDDING_MODEL, GENERATION_TEMPERATURE
from rag_tutoring.evaluate import Question, first_hit_rank
from rag_tutoring.generate import PROMPT_VERSION, Generator, UngroundedCitation

# Lexical proxy for "the answer signalled that the passages fall short". It is a
# keyword check, not a judgement: it will miss a hedge phrased differently and can
# fire on an answer that hedges about something irrelevant. Reported so a *trend*
# across runs is visible, and never as the finding -- the manual read of the misses
# and negatives is the finding. Keeping a weak measure is fine; mistaking it for a
# strong one is not.
_HEDGE_PHRASES = (
    "do not address",
    "don't address",
    "does not address",
    "doesn't address",
    "do not cover",
    "don't cover",
    "does not cover",
    "not directly",
    "only partly",
    "only partially",
    "no passage",
    "none of the passages",
    "not in these",
    "closest",
    "related",
)


@dataclass(frozen=True)
class GenerationOutcome:
    """One question's generation outcome. Carries **no answer text**.

    Same rule as ``QuestionResult``: an answer may quote its passages, chunk text is
    copyrighted, and one textbook's extraction carried the purchaser's email. So the
    committed artifact stays metadata, and the prose is written separately to a
    gitignored file for reading.
    """

    question_id: str
    style: str
    partition: str  # "hit" | "miss" | "negative"
    retrieval_rank: int | None
    answered: bool
    ungrounded: bool
    n_cited: int
    cited_a_labelled_page: bool
    hedge_phrase_present: bool
    answer_chars: int


@dataclass
class GenerationReport:
    provenance: dict
    outcomes: list[GenerationOutcome] = field(default_factory=list)
    # question_id -> answer text. Never serialised into the committed baseline.
    answers: dict[str, str] = field(default_factory=dict)

    def partition(self, name: str) -> list[GenerationOutcome]:
        return [o for o in self.outcomes if o.partition == name]

    def to_json(self) -> dict:
        """The committed artifact: provenance plus per-question metadata, no prose."""
        return {
            "provenance": self.provenance,
            "summary": {name: summarise(self.partition(name)) for name in PARTITIONS},
            "outcomes": [vars(o) for o in self.outcomes],
        }


PARTITIONS = ("hit", "miss", "negative")


def summarise(outcomes: Sequence[GenerationOutcome]) -> dict:
    """Mechanical rates over one partition. ``n`` is always reported alongside.

    The 07-30 lesson applied: a rate with no denominator invites being read as a
    result, and the negatives partition is n=5 -- nothing there is a finding in either
    direction, which the printed report says out loud.
    """
    n = len(outcomes)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "answered": sum(o.answered for o in outcomes),
        "ungrounded_citations": sum(o.ungrounded for o in outcomes),
        "uncited_answers": sum(o.answered and o.n_cited == 0 for o in outcomes),
        "cited_a_labelled_page": sum(o.cited_a_labelled_page for o in outcomes),
        "hedge_phrase_present": sum(o.hedge_phrase_present for o in outcomes),
        "mean_cited": round(sum(o.n_cited for o in outcomes) / n, 2),
        "mean_answer_chars": round(sum(o.answer_chars for o in outcomes) / n),
    }


def run(
    store,
    generator: Generator,
    questions: Sequence[Question],
    k: int = 5,
) -> GenerationReport:
    """Retrieve, generate, and score one pass over ``questions``.

    ``k`` defaults to 5 because that is what the product sends and what the retrieval
    baseline measures. Evaluating generation at a different ``k`` than the page uses
    would produce a number that describes a system nobody is running.
    """
    report = GenerationReport(
        provenance={
            "embedding_model": EMBEDDING_MODEL,
            "collection": store.collection_name,
            "chunks_indexed": store.count(),
            "documents_indexed": len(store.sources()),
            "generation_model": generator.model,
            "prompt_version": PROMPT_VERSION,
            "temperature": GENERATION_TEMPERATURE,
            "k": k,
        }
    )

    for question in questions:
        hits = store.query(question.question, k=k)
        citations = [cite(h) for h in hits]
        retrieved = [(h.source, h.page) for h in hits]
        rank = first_hit_rank(retrieved, question.sources)

        if question.is_negative:
            partition = "negative"
        elif rank is not None:
            partition = "hit"
        else:
            partition = "miss"

        answered = ungrounded = False
        cited: tuple[int, ...] = ()
        text = ""
        try:
            answer = generator.answer(question.question, citations)
            answered, cited, text = True, answer.cited, answer.text
        except UngroundedCitation:
            # Counted, not raised: how often the model cites a passage it was not
            # given is one of the things this eval exists to quantify.
            ungrounded = True

        # A lower bound, for the same reason recall is one: relevance judgements are
        # incomplete, so a cited passage that genuinely answers the question but was
        # never labelled counts as "not a labelled page" here.
        labelled = {(label.source, label.page) for label in question.sources}
        cited_a_labelled_page = any(retrieved[n - 1] in labelled for n in cited)

        report.outcomes.append(
            GenerationOutcome(
                question_id=question.id,
                style=question.style,
                partition=partition,
                retrieval_rank=rank,
                answered=answered,
                ungrounded=ungrounded,
                n_cited=len(cited),
                cited_a_labelled_page=cited_a_labelled_page,
                hedge_phrase_present=any(p in text.lower() for p in _HEDGE_PHRASES),
                answer_chars=len(text),
            )
        )
        if text:
            report.answers[question.id] = text

    return report


def format_report(report: GenerationReport) -> str:
    """A report that states its own limits, because the interesting n here is 5."""
    lines = ["Generation eval", "=" * 60, ""]
    p = report.provenance
    lines.append(
        f"  {p['generation_model']}  prompt v{p['prompt_version']}  temp {p['temperature']}"
    )
    lines.append(
        f"  over {p['documents_indexed']} documents / {p['chunks_indexed']:,} chunks, k={p['k']}"
    )
    lines.append("")

    header = (
        f"  {'partition':<10} {'n':>3} {'answered':>9} {'uncited':>8}"
        f" {'ungrounded':>11} {'cited-label':>12} {'hedged*':>8}"
    )
    lines += [header, "  " + "-" * (len(header) - 2)]
    for name in PARTITIONS:
        s = summarise(report.partition(name))
        if not s["n"]:
            continue
        lines.append(
            f"  {name:<10} {s['n']:>3} {s['answered']:>9} {s['uncited_answers']:>8}"
            f" {s['ungrounded_citations']:>11} {s['cited_a_labelled_page']:>12}"
            f" {s['hedge_phrase_present']:>8}"
        )

    lines += [
        "",
        "  hit      = retrieval put a labelled page in the top k",
        "  miss     = it did not; the model was handed five passages without one",
        "  negative = the corpus cannot answer the question at all",
        "",
        "  *hedged is a keyword proxy, not a judgement -- it can miss a hedge and can",
        "   fire on an irrelevant one. Read the miss and negative answers; there are",
        "   about 21 of them and they are short.",
        "  cited-label is a lower bound: relevance judgements are incomplete, so a",
        "   correct-but-unlabelled citation counts against this number.",
        "  The negative partition is n=5. Nothing there is a finding in either",
        "   direction -- there is no power at that size.",
    ]
    return "\n".join(lines)
