"""HTTP surface over retrieval: ask a question, get cited passages back.

**This endpoint retrieves; it does not answer.** It returns the passages that
best match the question, each with a citation, and leaves the reading to the
student. That is a deliberate stopping point, not an unfinished one: at the
current baseline a correct page is in the top 5 half the time but is the *top*
hit only 19% of the time (see ``eval/README.md``), so synthesising one confident
answer from the top hits would launder a retrieval miss into fluent prose a
student has no way to audit. Showing five sources and their scores puts the
judgement where the evidence supports it. A generation step, when it comes,
consumes this same response and needs nothing here to change.

Run it with::

    uvicorn rag_tutoring.api:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from rag_tutoring.citations import cite
from rag_tutoring.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, EMBEDDING_MODEL
from rag_tutoring.store import VectorStore

MAX_K = 20


class AskRequest(BaseModel):
    """A student's question.

    Both fields are bounded. ``k`` is the interesting one: it is a
    client-controlled multiplier on query cost, so leaving it open would let one
    caller ask for ten thousand neighbours. Twenty is well past what is useful to
    read -- recall stops improving long before then -- and this bound is the hook
    the Phase 4 rate limiting attaches to.
    """

    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=MAX_K)


class CitationOut(BaseModel):
    document: str
    page: int
    page_label: str
    source_type: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    citations: list[CitationOut]


class IndexInfo(BaseModel):
    """What was searched. Mirrors the provenance an eval report stamps.

    A result set means little without it: the same question against a different
    chunk budget or embedding model is a different system. Serving it here means
    a student's transcript and a benchmark run can be traced to the same index.
    """

    embedding_model: str
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    collection: str
    chunks_indexed: int
    documents_indexed: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and the index once, at startup.

    Constructing ``VectorStore`` loads sentence-transformers weights and opens
    Chroma -- seconds, not milliseconds. Doing it per request would pay that on
    every question; doing it at import would make this module unimportable
    without an index, which is what keeps the test suite model-free.

    ``documents_indexed`` is counted here rather than per request because
    ``sources()`` scans every chunk's metadata. The cost is the staleness: rebuild
    the index under a running server and this number describes the old one. For a
    pilot, restarting after a rebuild is the honest fix -- invalidating it
    properly is the same cache-invalidation question Phase 4 takes up.
    """
    store = VectorStore()
    app.state.store = store
    app.state.index_info = IndexInfo(
        embedding_model=EMBEDDING_MODEL,
        chunk_max_tokens=CHUNK_MAX_TOKENS,
        chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
        collection=store.collection_name,
        chunks_indexed=store.count(),
        documents_indexed=len(store.sources()),
    )
    yield
    app.state.store = None


app = FastAPI(
    title="RAG Tutoring",
    description="Retrieval over deep learning tutoring materials, with citations.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_store(request: Request) -> VectorStore:
    """Hand the request the process-wide store.

    A dependency rather than a module-level global so tests can substitute a
    stub via ``app.dependency_overrides`` and run without an index or a model.
    """
    return request.app.state.store


StoreDep = Annotated[VectorStore, Depends(get_store)]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, store: StoreDep) -> AskResponse:
    """Return the passages most relevant to ``question``, best first.

    Responses carry verbatim source text, which is copyrighted material, so
    nothing here logs a response body -- the same reason ``eval/baseline.json``
    stores scores and citations but never the retrieved passage.
    """
    hits = store.query(payload.question, k=payload.k)
    return AskResponse(
        question=payload.question,
        citations=[CitationOut(**asdict(cite(h))) for h in hits],
    )


@app.get("/health", response_model=IndexInfo)
def health(request: Request) -> IndexInfo:
    """What this process is serving.

    Deliberately reads the values stamped at startup instead of re-querying, so
    a load balancer polling this endpoint cannot make it expensive.
    """
    return request.app.state.index_info
