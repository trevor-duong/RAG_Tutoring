"""Tests for the HTTP surface.

These run with **no index and no embedding model**, the same constraint the rest
of the suite holds to. Two things make that work, and both are load-bearing:

* the store arrives through a dependency, so a stub can replace it;
* ``TestClient(app)`` is used bare rather than as a context manager, because
  entering the context runs ``lifespan``, and ``lifespan`` constructs the real
  ``VectorStore`` -- which loads sentence-transformers weights and opens Chroma.

So a test that needs startup state sets it on ``app.state`` explicitly. That is
also the honest shape of the coverage: dependency-overridden tests never
exercise the real wiring, which has to be checked by starting the server.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from rag_tutoring import generate
from rag_tutoring.api import (
    INDEX_HTML,
    MAX_K,
    AnswerOut,
    AskResponse,
    IndexInfo,
    app,
    get_generator,
    get_store,
)
from rag_tutoring.generate import Answer, TruncatedAnswer, UngroundedCitation


def _script_of(page: str) -> str:
    """The page's JavaScript, with ``//`` comments removed.

    Assertions about what the frontend *does* have to read the code, not the
    comments explaining the code -- a comment warning against a pattern contains
    that pattern. Splitting on ``//`` also truncates any URL on a code line, which
    is harmless here and would matter if this were ever used for more than
    substring checks.

    The single-block assertion is the point: this returns *a* script, and adding a
    second one would silently shrink what every caller checks while leaving them
    green. Splitting the page's JavaScript is a fine thing to do -- it just has to
    be done here too, deliberately, rather than discovered later.
    """
    blocks = page.split("<script>")[1:]
    assert len(blocks) == 1, "the page has more than one script block; widen this helper"
    body = blocks[0].split("</script>")[0]
    return "\n".join(line.split("//")[0] for line in body.splitlines())


@dataclass(frozen=True)
class FakeRetrieved:
    text: str
    source: str
    source_type: str
    page: int
    score: float


class FakeStore:
    """Returns ``k`` synthetic hits in descending score order, and records the
    ``k`` it was asked for so a test can prove the parameter is not ignored."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def query(self, text: str, k: int = 5) -> list[FakeRetrieved]:
        self.calls.append((text, k))
        return [
            FakeRetrieved(
                text=f"passage {i}",
                source="GloVe- Global Vectors for Word Representation",
                source_type="paper",
                page=100 + i,
                score=1.0 - i / 10,
            )
            for i in range(k)
        ]


@pytest.fixture
def store():
    fake = FakeStore()
    app.dependency_overrides[get_store] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def stamped_index_info():
    """Set startup state for the duration of one test, then put it back.

    ``app`` is module-global, so assigning ``app.state.index_info`` and walking
    away leaks into every later test. Nothing else reads it today, which is
    exactly what would make the leak invisible until something did.
    """
    sentinel = object()
    previous = getattr(app.state, "index_info", sentinel)
    yield IndexInfo(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_max_tokens=240,
        chunk_overlap_tokens=60,
        collection="tutoring-all-MiniLM-L6-v2",
        chunks_indexed=9169,
        documents_indexed=37,
        generation_model="claude-sonnet-5",
    )
    if previous is sentinel:
        del app.state.index_info
    else:
        app.state.index_info = previous


def test_ask_returns_citations_best_first(client, store):
    body = client.post("/ask", json={"question": "what is a word embedding?", "k": 3}).json()
    assert body["question"] == "what is a word embedding?"
    assert len(body["citations"]) == 3, "every retrieved hit must reach the response"
    scores = [c["score"] for c in body["citations"]]
    assert scores == sorted(scores, reverse=True), "retrieval order must survive formatting"


def test_ask_passes_k_through_to_the_store(client, store):
    """Asserting only on the response length would pass with ``k`` hardcoded.

    The stub returns ``k`` hits, so a route that ignored the parameter and always
    asked for 5 would still return a plausible-looking list. Checking what the
    store was actually called with is what makes this able to fail.
    """
    client.post("/ask", json={"question": "what is dropout?", "k": 7})
    assert store.calls == [("what is dropout?", 7)]


def test_ask_defaults_to_five(client, store):
    client.post("/ask", json={"question": "what is dropout?"})
    assert store.calls[0][1] == 5


def test_response_carries_the_reader_facing_citation(client, store):
    citation = client.post("/ask", json={"question": "what is glove?"}).json()["citations"][0]
    assert citation["document"] == "GloVe: Global Vectors for Word Representation"
    assert citation["page_label"] == "PDF page 100"
    assert citation["source_type"] == "paper"
    assert citation["snippet"] == "passage 0"


@pytest.mark.parametrize("k", [0, -1, MAX_K + 1])
def test_k_outside_the_bound_is_rejected(client, store, k):
    """``k`` is a client-controlled multiplier on query cost, so it is bounded."""
    assert client.post("/ask", json={"question": "valid question", "k": k}).status_code == 422
    assert store.calls == [], "a rejected request must not reach the store"


@pytest.mark.parametrize("question", ["", "ab"])
def test_too_short_a_question_is_rejected(client, store, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_an_overlong_question_is_rejected(client, store):
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


class FakeGenerator:
    """Stands in for the generation layer. ``model`` matches the real attribute so a
    test can assert what ``/health`` would report."""

    model = "fake-model"

    def __init__(self, answer=None, raises=None) -> None:
        self.answer_value = answer
        self.raises = raises
        self.calls: list[tuple[str, int]] = []

    def answer(self, question: str, citations):
        self.calls.append((question, len(citations)))
        if self.raises is not None:
            raise self.raises
        return self.answer_value


@pytest.fixture
def generator():
    """Install a generator for one test, then remove it.

    Yields a setter rather than a generator, because different tests need different
    behaviour from it. Removes only its own override rather than clearing the dict, so
    it cannot quietly undo the ``store`` fixture's setup during teardown.
    """

    def install(fake: FakeGenerator) -> FakeGenerator:
        app.dependency_overrides[get_generator] = lambda: fake
        return fake

    yield install
    app.dependency_overrides.pop(get_generator, None)


def test_generation_off_returns_passages_with_no_answer(client, store):
    """The default, and a working product rather than a degraded one.

    No generator is installed, so this is also the state every other test in this file
    runs in -- which is why the answer field has to be optional rather than absent.
    """
    body = client.post("/ask", json={"question": "what is dropout?"}).json()
    assert body["answer"] is None
    assert len(body["citations"]) == 5, "passages must survive generation being off"


def test_an_answer_is_returned_alongside_its_sources(client, store, generator):
    """The shape the whole design rests on: never the answer instead of the passages."""
    generator(
        FakeGenerator(
            answer=Answer(
                text="Your model is overfitting. [source 1]",
                cited=(1,),
                model="fake-model",
                prompt_version="1",
            )
        )
    )
    body = client.post("/ask", json={"question": "why is validation error rising?"}).json()
    assert body["answer"]["text"] == "Your model is overfitting. [source 1]"
    assert body["answer"]["cited"] == [1]
    assert body["answer"]["prompt_version"] == "1"
    assert len(body["citations"]) == 5, "the answer must not displace the passages"


def test_the_generator_receives_the_retrieved_passages(client, store, generator):
    """Asserting only that an answer came back would pass if the generator were handed
    an empty list and the stub returned its canned text anyway."""
    fake = generator(
        FakeGenerator(answer=Answer(text="ok", cited=(), model="fake-model", prompt_version="1"))
    )
    client.post("/ask", json={"question": "what is dropout?", "k": 3})
    assert fake.calls == [("what is dropout?", 3)]


def test_an_ungrounded_answer_is_dropped_and_the_passages_still_serve(
    client, store, generator, caplog
):
    """A citation pointing at a passage that was never retrieved must not reach a
    student, and the drop must be visible in the log -- from the outside this response
    is indistinguishable from generation simply being switched off.
    """
    generator(FakeGenerator(raises=UngroundedCitation("cites [source 9] of 5")))
    with caplog.at_level(logging.WARNING):
        body = client.post("/ask", json={"question": "what is dropout?"}).json()
    assert body["answer"] is None
    assert len(body["citations"]) == 5
    assert any("UngroundedCitation" in r.getMessage() for r in caplog.records), (
        "a silently dropped answer looks identical to generation being off"
    )


def test_a_truncated_answer_is_dropped_and_the_passages_still_serve(
    client, store, generator, caplog
):
    """The second reason an answer can be dropped, and it must be distinguishable.

    Both failures leave ``answer`` null, so the log line is the only thing that says
    which happened -- and they need different responses: an ungrounded citation is a
    prompt problem, a truncation is a budget one. A log that said only "dropped an
    answer" would make a rising rate uninvestigable.
    """
    generator(FakeGenerator(raises=TruncatedAnswer("hit the 2000-token budget")))
    with caplog.at_level(logging.WARNING):
        body = client.post("/ask", json={"question": "what is dropout?"}).json()
    assert body["answer"] is None
    assert len(body["citations"]) == 5
    assert any("TruncatedAnswer" in r.getMessage() for r in caplog.records)


def test_a_generation_failure_does_not_take_retrieval_down_with_it(client, store, generator):
    """Only ``UngroundedCitation`` and ``TruncatedAnswer`` are handled, so this pins the
    current contract: any other failure propagates. Worth stating explicitly -- an API
    timeout reaching a student as a 500 is a decision, and this is where to revisit
    it."""
    generator(FakeGenerator(raises=RuntimeError("API unreachable")))
    with pytest.raises(RuntimeError):
        client.post("/ask", json={"question": "what is dropout?"})


def test_the_page_is_served_from_packaged_data(client):
    """``INDEX_HTML`` is resolved from the module, so this fails if the file moves.

    Asserting on the status code alone would be close to unfailable. The header
    check is what distinguishes "served the page" from "served something".
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_the_page_posts_to_the_endpoint_it_is_served_with():
    """Pin the one contract that spans the two halves of this phase.

    The page hardcodes its own API paths, so renaming a route leaves the tests
    above passing and the UI silently broken -- the request 404s in a browser
    nobody is watching during a test run. Reading the served page and checking it
    references the routes this module actually registers is what makes that
    failure visible here instead of in front of a student.
    """
    script = _script_of(INDEX_HTML.read_text())
    registered = {route.path for route in app.routes}
    for call in ('fetch("/ask"', 'fetch("/health")'):
        assert call in script, f"the page no longer calls {call}"
    assert {"/ask", "/health", "/"} <= registered


def test_the_page_and_the_server_agree_on_the_citation_marker():
    """Pin the marker format across the two files that both have to know it.

    ``generate.py`` chose ``[source N]`` over ``[N]`` for a measured reason -- bare
    bracketed numbers appear 1,342 times in the corpus. If either side changes format
    alone, every citation renders as literal ``[source 3]`` inside the prose: no error,
    no failing Python test, just an answer whose audit trail stopped working.
    """
    script = _script_of(INDEX_HTML.read_text())
    match = re.search(r"const MARKER = /(.+?)/[a-z]*;", script)
    assert match, "the page no longer defines a citation marker pattern"
    page_marker = re.compile(match.group(1), re.IGNORECASE)

    sample = "co-adaptation is prevented [source 3]"
    assert page_marker.search(sample), "the page cannot find a marker the server emits"
    assert generate._MARKER.search(sample), "the server no longer emits what it documents"
    assert page_marker.search(sample).group(1) == "3", "the passage number must be captured"

    # The collision the format exists to avoid, asserted on both sides.
    assert not page_marker.search("prevents co-adaptation [15]")
    assert not generate._MARKER.search("prevents co-adaptation [15]")


def test_the_page_reads_the_answer_fields_the_response_actually_carries():
    """One more field crossing the frontend/backend boundary, pinned the same way.

    Rename ``answer`` to ``generated`` in ``AskResponse`` and every Python test still
    passes while the page silently stops showing answers -- the response is still valid
    JSON, the citations still render, and nothing errors. Checking the page reads the
    names the model actually serializes is what makes that visible here.
    """
    script = _script_of(INDEX_HTML.read_text())
    served = set(AskResponse.model_fields) | set(AnswerOut.model_fields)
    assert {"answer", "citations"} <= served, "the response no longer has these fields"
    assert "data.answer" in script, "the page no longer reads the answer off the response"
    for field in ("answer.text", "answer.model"):
        assert field in script, f"the page no longer renders {field}"


def test_the_page_does_not_parse_model_output_as_markup():
    """Answer text is model output containing text extracted from PDFs. Assembling it
    into an HTML string is the one shortcut that would turn that into a rendering
    bug at best, and injected markup at worst."""
    script = _script_of(INDEX_HTML.read_text())
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script


def test_the_page_never_touches_the_raw_page_index():
    """The frontend must render ``page_label``, never compose a reference itself.

    ``citations.py`` exists because ``Chunk.page`` is pypdf's index, not the
    number printed on the page. A template literal like ``p. ${c.page}`` in the
    browser would reintroduce exactly that misdirection, and no test of the Python
    would catch it, because the defect would live entirely in the frontend.

    Scoped to the script and with comments stripped, so the assertion is about
    what the page executes rather than what it says about itself -- the prose
    above the offending line necessarily names the pattern it is warning against.
    """
    script = _script_of(INDEX_HTML.read_text())
    assert "page_label" in script, "the label must reach the reader"
    assert not re.search(r"c\.page\b", script), "the raw index must not be rendered"
    assert "p. " not in script, "a hardcoded 'p. ' is the citation bug this guards"


def test_health_reports_what_is_actually_indexed(client, stamped_index_info):
    """Served from startup-stamped state, so no store override is needed -- and
    none would help, since the route reads ``app.state`` rather than a dependency."""
    app.state.index_info = stamped_index_info
    body = client.get("/health").json()
    assert body["chunks_indexed"] == 9169
    assert body["documents_indexed"] == 37
    assert body["collection"] == "tutoring-all-MiniLM-L6-v2"
    assert body["generation_model"] == "claude-sonnet-5", (
        "which model wrote an answer is provenance, not something to infer from"
        " whether prose happens to be present"
    )
