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

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from rag_tutoring.api import MAX_K, IndexInfo, app, get_store


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


def test_health_reports_what_is_actually_indexed(client, stamped_index_info):
    """Served from startup-stamped state, so no store override is needed -- and
    none would help, since the route reads ``app.state`` rather than a dependency."""
    app.state.index_info = stamped_index_info
    body = client.get("/health").json()
    assert body["chunks_indexed"] == 9169
    assert body["documents_indexed"] == 37
    assert body["collection"] == "tutoring-all-MiniLM-L6-v2"
