"""Tests for the generation layer.

These run with **no API key and no network**, the same constraint that keeps the rest
of the suite free of an index and an embedding model. The client is injected, so a
stub stands in for it; nothing here constructs an ``anthropic.Anthropic`` except the
one test that checks the factory, and that call opens no connection.

The assertions worth their place are the citation-validation ones. Prompting a model
to cite and then trusting the markers is a check that cannot fail, so the tests below
are mostly about what happens when the model cites something it was not given.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from rag_tutoring.citations import Citation
from rag_tutoring.generate import (
    PROMPT_VERSION,
    Answer,
    Generator,
    TruncatedAnswer,
    UngroundedCitation,
    cited_sources,
    format_passages,
    generator_from_env,
)

PASSAGE = Citation(
    document="Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    page=17,
    page_label="PDF page 17",
    source_type="paper",
    score=0.4881,
    snippet="architecture. No input dropout was used.",
)


def passages(n: int) -> list[Citation]:
    return [replace(PASSAGE, page=100 + i, page_label=f"PDF page {100 + i}") for i in range(n)]


@dataclass(frozen=True)
class FakeBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class FakeResponse:
    content: list[FakeBlock]
    # Defaults to the happy path so the existing tests keep asserting it. Present at
    # all because the real response carries it and ``answer`` now reads it -- a stub
    # missing a field the code under test uses is a stub for a different object.
    stop_reason: str = "end_turn"


class FakeMessages:
    def __init__(self, blocks: list[FakeBlock], stop_reason: str = "end_turn") -> None:
        self.blocks = blocks
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(content=self.blocks, stop_reason=self.stop_reason)


class FakeClient:
    """Records what it was asked for, so a test can prove the prompt was sent."""

    def __init__(
        self,
        text: str = "Overfitting. [source 1]",
        blocks=None,
        stop_reason: str = "end_turn",
    ) -> None:
        self.messages = FakeMessages(
            blocks if blocks is not None else [FakeBlock(text)], stop_reason
        )


# --- passage formatting ----------------------------------------------------------


def test_passages_are_numbered_from_one():
    """The model cites by these numbers, so an off-by-one here misattributes every
    claim in the answer while leaving it looking perfectly well cited."""
    out = format_passages(passages(3))
    assert "[source 1]" in out
    assert "[source 3]" in out
    assert "[source 0]" not in out
    assert "[source 4]" not in out


def test_the_model_is_given_the_page_label_not_the_raw_index():
    """Same rule as the frontend: what reaches a reader says which numbering it means.

    If the model were handed a bare page number it would write "p. 100" into prose
    that a student then trusts -- reintroducing the front-matter offset bug one layer
    further from where ``citations.py`` guards against it.
    """
    out = format_passages(passages(1))
    assert "PDF page 100" in out
    assert "p. 100" not in out


# --- citation validation --------------------------------------------------------


def test_cited_sources_records_first_appearance_order_without_duplicates():
    assert cited_sources("a [source 2] b [source 1] c [source 2]", available=3) == (2, 1)


def test_a_citation_out_of_range_is_rejected():
    """The load-bearing assertion. An answer citing a source it was never given is
    prose that reads as grounded and points at nothing."""
    with pytest.raises(UngroundedCitation):
        cited_sources("claim [source 6]", available=5)


def test_zero_is_out_of_range():
    """Passages are 1-based, so [source 0] names nothing -- and would silently become
    a valid Python index if this were ever used to subscript the list."""
    with pytest.raises(UngroundedCitation):
        cited_sources("claim [source 0]", available=5)


def test_bare_bracket_numbers_from_the_corpus_are_not_treated_as_citations():
    """The measured reason the marker is ``[source N]`` and not ``[N]``.

    Bare bracketed numbers occur 1,342 times across 11.6% of the corpus's pages,
    because papers cite by number. If a quoted passage brought "[15]" into the answer,
    a bare-marker scheme would either reject a perfectly good answer or -- with k
    above 15 -- silently accept it as a citation of an unrelated source.
    """
    text = "Dropout prevents co-adaptation [15], as shown by Srivastava et al. [source 2]"
    assert cited_sources(text, available=5) == (2,)


def test_marker_matching_is_case_insensitive():
    assert cited_sources("claim [Source 3]", available=5) == (3,)


def test_an_uncited_answer_is_allowed_through_validation():
    """Validation's job is that citations point at real passages, not that they exist.

    An answer with no markers is a *quality* problem, and quality is what the eval
    measures over 37 questions -- turning it into a hard error here would mean one
    unhelpful answer takes down the request instead of being counted.
    """
    assert cited_sources("The passages do not address this directly.", available=5) == ()


# --- the generator --------------------------------------------------------------


def test_answer_carries_the_provenance_needed_to_reproduce_it():
    gen = Generator(FakeClient("Overfitting. [source 2]"), model="test-model")
    answer = gen.answer("why does validation error rise?", passages(3))
    assert isinstance(answer, Answer)
    assert answer.text == "Overfitting. [source 2]"
    assert answer.cited == (2,)
    assert answer.model == "test-model"
    assert answer.prompt_version == PROMPT_VERSION


def test_the_request_carries_the_question_the_passages_and_the_grounding_rules():
    """Asserting only on the returned text would pass with an empty prompt.

    The stub returns its canned answer no matter what it is sent, so what makes this
    able to fail is checking what the client was actually called with.
    """
    client = FakeClient()
    Generator(client, model="test-model", temperature=None, max_tokens=700).answer(
        "what is dropout?", passages(2)
    )
    (call,) = client.messages.calls
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 700
    assert "only the numbered passages" in call["system"]
    assert "do not fill" in call["system"].lower(), "the no-invention rule must be sent"
    content = call["messages"][0]["content"]
    assert "what is dropout?" in content
    assert "[source 2]" in content, "every retrieved passage must reach the model"


def test_no_temperature_key_is_sent_when_it_is_unset():
    """The regression this exists for actually happened.

    ``claude-sonnet-5`` rejects the *presence* of ``temperature`` with a 400, not a
    particular value, so sending ``temperature=None`` would fail exactly as sending
    ``0.0`` did. Asserting the key is absent -- rather than that its value is None --
    is the difference between a test that catches the 400 and one that passes while
    every live request fails.
    """
    client = FakeClient()
    Generator(client, temperature=None).answer("what is dropout?", passages(2))
    (call,) = client.messages.calls
    assert "temperature" not in call
    assert "top_p" not in call, "the same deprecation covers top_p"


def test_a_temperature_is_still_sent_when_one_is_configured():
    """The knob is kept, not deleted: an older model still accepts it, and a config
    that silently ignores its own value is worse than not having the setting."""
    client = FakeClient()
    Generator(client, temperature=0.0).answer("what is dropout?", passages(2))
    (call,) = client.messages.calls
    assert call["temperature"] == 0.0


def test_generating_with_no_passages_is_refused():
    """Nothing to ground an answer in, so asking anyway invites the unsourced answer
    this module exists to prevent."""
    with pytest.raises(ValueError):
        Generator(FakeClient()).answer("what is dropout?", [])


def test_an_answer_cut_off_by_the_output_budget_is_dropped():
    """Measured, not hypothesised: 2 of 37 answers in the 2026-08-05 eval were cut
    mid-word, because sonnet-5's thinking blocks are billed against the same
    ``max_tokens`` ceiling as the visible text.

    Dropping rather than serving is about *where* the cut lands. The prompt asks the
    model to state what the passages do not cover, and that qualification comes last,
    so truncation removes the hedge and leaves the confident part -- a shorter, more
    certain answer than the model actually produced.
    """
    client = FakeClient(
        "Batch norm fixes covariate shift, but they do not confir", stop_reason="max_tokens"
    )
    with pytest.raises(TruncatedAnswer):
        Generator(client).answer("what is internal covariate shift?", passages(5))


def test_truncation_is_checked_before_the_citations_are_parsed():
    """A half-written marker must report as truncation, not as a citation problem.

    ``[source 9]`` here is out of range and would raise ``UngroundedCitation`` if
    validation ran first -- which would send a prompt-quality alert for what is
    actually a budget problem, and hide the real cause.
    """
    client = FakeClient("Overfitting [source 9]", stop_reason="max_tokens")
    with pytest.raises(TruncatedAnswer):
        Generator(client).answer("why?", passages(3))


def test_an_ungrounded_citation_propagates_rather_than_being_repaired():
    gen = Generator(FakeClient("Overfitting. [source 9]"))
    with pytest.raises(UngroundedCitation):
        gen.answer("why?", passages(3))


def test_only_text_blocks_reach_the_answer():
    """A non-text block must not crash the join or leak into prose a student reads."""
    client = FakeClient(
        blocks=[
            FakeBlock(text="ignored", type="thinking"),
            FakeBlock(text="Overfitting. [source 1]"),
        ]
    )
    assert Generator(client).answer("why?", passages(2)).text == "Overfitting. [source 1]"


# --- the env factory ------------------------------------------------------------


def test_no_key_means_generation_is_off_rather_than_broken(monkeypatch):
    """The property that keeps the key optional: no key is a working retrieval-only
    product, which is exactly what this served before generation existed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generator_from_env() is None


def test_the_dotenv_is_read_only_at_entry_points():
    """Guards the property every other test in this file depends on: the suite runs
    with no key.

    ``load_dotenv`` belongs to the process entry points -- the API's lifespan and the
    generation eval script. Move it into a library module (``config.py`` is the tempting
    place, since everything imports it) and importing anything would hand pytest
    whatever key sits in the developer's ``.env``. Every no-key assertion here would
    then pass or fail depending on a gitignored file, which is a suite that cannot fail
    for the one reason it exists.

    A grep rather than a behavioural check on purpose: the failure is a *new call site*
    someone adds later, and no runtime assertion sees a call that has not been written.
    """
    root = Path(__file__).resolve().parents[1]
    allowed = {"src/rag_tutoring/api.py", "scripts/run_generation_eval.py"}
    found = {
        str(path.relative_to(root))
        for directory in ("src", "scripts")
        for path in (root / directory).rglob("*.py")
        if "load_dotenv" in path.read_text()
    }
    assert found == allowed, f"unexpected .env load sites: {sorted(found - allowed)}"


def test_a_key_produces_a_generator(monkeypatch):
    """Constructing the client opens no connection, so this stays network-free.

    The placeholder deliberately does not use the real ``sk-ant-`` prefix: the factory
    only checks that the variable is non-empty, and a realistic-looking key in a
    committed file is the kind of thing secret scanners flag and humans then learn to
    ignore. Same reasoning as the synthetic ``example.*`` email fixtures.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    gen = generator_from_env()
    assert isinstance(gen, Generator)
    assert gen.model, "the configured model must be carried, not left empty"
