"""HTTP surface over retrieval: ask a question, get an answer *and* its sources.

**The answer never replaces the passages.** Both are returned, and the page renders
the passages underneath the answer. That shape is the design, and it comes straight
out of the baseline: a correct page is in the top 5 half the time but is the *top* hit
only 19% of the time (see ``eval/README.md``), so an answer synthesised from these
passages is sometimes synthesised from the wrong ones. Returning the sources it was
built from is what makes that visible to a student instead of hidden behind fluent
prose. Answer-only would be the one variant the evidence does not support.

Generation is **optional and off by default**. With no ``ANTHROPIC_API_KEY`` the
endpoint serves passages alone, which is a working product rather than a degraded one
-- it is exactly what this served before ``generate.py`` existed. That is also what
keeps the test suite free of a key, and the whole suite runs with no index, no
embedding model and no API key.

Two failures drop the answer and serve the passages alone: a citation to a passage the
model was not given, and an answer cut off by the output budget. Neither is repaired --
see ``generate.UngroundedCitation`` and ``generate.TruncatedAnswer`` for why. Both leave
``answer`` null, which is a state the page already renders.

Run it with::

    uvicorn rag_tutoring.api:app --reload

That serves the JSON API and, at ``/``, the one static page that consumes it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_tutoring.citations import cite
from rag_tutoring.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    EMBEDDING_MODEL,
    ENV_FILE,
)
from rag_tutoring.generate import (
    Generator,
    TruncatedAnswer,
    UngroundedCitation,
    generator_from_env,
)
from rag_tutoring.store import VectorStore

log = logging.getLogger(__name__)

MAX_K = 20

# Resolved from this module rather than the working directory, so the page is found
# whether the server is started from the repo root or anywhere else. It travels with
# the package as data -- declared in ``pyproject.toml`` under ``package-data``, where
# the comment explains what that declaration does and does not buy.
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


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


class AnswerOut(BaseModel):
    """The synthesised answer, and which passages it claims to rest on.

    ``cited`` is 1-based into ``AskResponse.citations`` and is already validated --
    every number in it names a passage in the same response. A client can render the
    markers as links without re-checking, which is the point of validating server-side
    rather than asking the page to be careful.
    """

    text: str
    cited: list[int]
    model: str
    prompt_version: str


class AskResponse(BaseModel):
    """Answer plus sources. ``answer`` is ``None`` when generation is off or when its
    citations failed validation; the passages are always present either way."""

    question: str
    answer: AnswerOut | None
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
    # How many of those a question can actually reach. The two differ because a
    # document's apparatus -- reference lists, acknowledgments, contents pages -- is
    # indexed and then excluded at query time. Reporting only the total would state
    # that 9,169 chunks are searchable while 782 of them are unreachable, which is
    # the same class of untruth as a config documenting a mechanism that is not wired
    # up: it reads as fact and nothing contradicts it.
    chunks_retrievable: int
    documents_indexed: int
    # None when no key is configured. Reported because "did this answer come from a
    # model, and which one?" is not something a student or a saved transcript should
    # have to infer from whether prose happens to be present.
    generation_model: str | None


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
    # A local .env is read *here*, at the process entry point, rather than at import of
    # config -- see the ENV_FILE comment. ``override=False`` is the default and the one
    # that matters: a real environment variable, which is how a deployment supplies the
    # key, always beats a stale file someone left in the working copy.
    load_dotenv(ENV_FILE)

    store = VectorStore()
    generator = generator_from_env()
    app.state.store = store
    app.state.generator = generator
    app.state.index_info = IndexInfo(
        embedding_model=EMBEDDING_MODEL,
        chunk_max_tokens=CHUNK_MAX_TOKENS,
        chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
        collection=store.collection_name,
        chunks_indexed=store.count(),
        chunks_retrievable=store.count(include_structural=False),
        documents_indexed=len(store.sources()),
        generation_model=generator.model if generator else None,
    )
    if generator is None:
        # WARNING rather than INFO on purpose: uvicorn's default log config does not
        # enable INFO for module loggers, so an info-level line here would never be
        # printed -- a startup notice that cannot appear is no notice at all. Only the
        # surprising state is logged; ``/health`` is the authoritative answer for both,
        # and it does not depend on how logging happens to be configured.
        log.warning("generation is off: ANTHROPIC_API_KEY is not set; serving passages only")
    yield
    app.state.store = None
    app.state.generator = None


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


def get_generator(request: Request) -> Generator | None:
    """Hand the request the process-wide generator, or ``None`` if generation is off.

    A dependency for the same reason the store is one: a test overrides it with a stub
    and never needs a key. ``None`` is a legitimate value here, not a missing
    dependency -- see the module docstring.
    """
    return getattr(request.app.state, "generator", None)


StoreDep = Annotated[VectorStore, Depends(get_store)]
GeneratorDep = Annotated["Generator | None", Depends(get_generator)]


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, store: StoreDep, generator: GeneratorDep) -> AskResponse:
    """Answer ``question`` from the passages most relevant to it, best first.

    Responses carry verbatim source text, so nothing here logs a response body -- the
    same reason ``eval/baseline.json`` stores scores and citations but never the
    retrieved passage. The ungrounded-citation warning below is deliberately built
    from counts only, so it stays loggable.
    """
    hits = store.query(payload.question, k=payload.k)
    citations = [cite(h) for h in hits]

    answer = None
    if generator is not None and citations:
        try:
            generated = generator.answer(payload.question, citations)
            answer = AnswerOut(**asdict(generated))
        except (UngroundedCitation, TruncatedAnswer) as exc:
            # Serve the passages alone rather than prose citing a source that is not
            # there, or prose whose qualifying tail was cut off. Logged because a rising
            # rate here is a prompt or budget problem, and it is invisible from the
            # outside -- the response looks exactly like generation being off.
            log.warning("dropped an answer: %s: %s", type(exc).__name__, exc)

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=[CitationOut(**asdict(c)) for c in citations],
    )


@app.get("/health", response_model=IndexInfo)
def health(request: Request) -> IndexInfo:
    """What this process is serving.

    Deliberately reads the values stamped at startup instead of re-querying, so
    a load balancer polling this endpoint cannot make it expensive.
    """
    return request.app.state.index_info


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the one page that consumes this API.

    A single self-contained file served by the same process, rather than a
    separate frontend: it means one deployable artifact, no build step, and no
    CORS surface at all, because the page and the API share an origin. A
    component framework would buy routing and reuse that one page has no use for.

    Kept out of the OpenAPI schema -- it is the UI, not an operation a client
    calls.
    """
    return FileResponse(INDEX_HTML)
