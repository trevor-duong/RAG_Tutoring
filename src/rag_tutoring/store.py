"""A thin vector store over Chroma, exposing only add / query / count.

The rest of the codebase talks to this interface, never to Chroma directly.
Per ADR 0001 this is the seam that keeps the store swappable: replacing Chroma
with pgvector later should touch this file and nothing that calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag_tutoring.config import CHROMA_DIR, EMBEDDING_MODEL, MODEL_MAX_TOKENS
from rag_tutoring.ingest import Chunk


@dataclass(frozen=True)
class Retrieved:
    """One search hit: the chunk text, where it came from, and how close it was."""

    text: str
    source: str
    source_type: str
    page: int
    score: float  # cosine similarity in [0, 1]; higher is more relevant
    # Whether this chunk is a document's apparatus rather than its exposition. Only
    # ever True when a caller asked for structural chunks; the default query path
    # excludes them. Carried so the eval can count how many top-k slots they take.
    structural: bool = False


class VectorStore:
    """Embeds chunks locally and stores them in a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        model_name: str = EMBEDDING_MODEL,
        collection: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        # Chunking sizes its windows against config.MODEL_MAX_TOKENS, but the
        # limit that actually binds is the model's own. If the two disagree --
        # after swapping to a model with a different window, say -- chunks are
        # built to the wrong budget and get silently truncated again. Checking
        # the constant against reality is what keeps the config from lying.
        if self.model.max_seq_length != MODEL_MAX_TOKENS:
            raise ValueError(
                f"{model_name} accepts {self.model.max_seq_length} word-pieces but "
                f"config.MODEL_MAX_TOKENS is {MODEL_MAX_TOKENS}; update the config "
                f"(and re-chunk) before indexing"
            )
        # Version the collection by model: embeddings from different models are
        # not comparable, so each model gets its own index.
        slug = model_name.rsplit("/", 1)[-1].replace(".", "-")
        self.collection_name = collection or f"tutoring-{slug}"
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
        )

    def reset(self) -> None:
        """Drop and recreate the collection.

        Upsert keys on chunk id (source::page::index), so re-indexing after a
        chunk-size change overwrites matching ids but strands the extra chunks
        the previous, differently-sized run produced. Chroma persists across
        kernel restarts, so those orphans would linger and pollute retrieval.
        Resetting first makes re-indexing a clean rebuild -- cheap here because
        embeddings are local (see ADR 0001).
        """
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    def _reject_oversized(self, chunks: list[Chunk]) -> None:
        """Fail loudly on chunks the model would silently truncate.

        This is the guard for the failure mode that hid here for a week: input
        past the model's window is dropped without warning, so the stored text
        and the vector it was matched on quietly diverge -- a chunk gets cited
        to a student on the strength of a passage the embedding never saw.
        Nothing raises, no test goes red, and retrieval still looks plausible.

        The check lives at the store boundary rather than in chunking because
        this is where the model's limit actually binds, so it covers any future
        source of chunks, not just ``chunk_pdf``.
        """
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length
        oversized = [
            (c.id, n)
            for c in chunks
            if (n := len(tokenizer.encode(c.text, add_special_tokens=True))) > limit
        ]
        if oversized:
            listed = ", ".join(f"{cid} ({n} > {limit})" for cid, n in oversized[:3])
            raise ValueError(
                f"{len(oversized)} chunk(s) exceed the {limit} word-piece embedding "
                f"window and would be silently truncated: {listed}"
                + ("..." if len(oversized) > 3 else "")
            )

    def _batch_size(self) -> int:
        """Largest upsert Chroma will accept in one call.

        Chroma caps a single write (5,461 on this build -- it comes from
        SQLite's bound-variable limit, not from anything about embeddings). The
        full corpus is ~8k chunks, so ingestion has to be chunked into batches
        or the write fails outright.
        """
        get_max = getattr(self._client, "get_max_batch_size", None)
        return get_max() if callable(get_max) else getattr(self._client, "max_batch_size", 1000)

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks, in batches the backend will accept.

        Upsert (not add) so re-running an ingestion cell overwrites by id
        instead of erroring on duplicates.

        Batching is not atomic: a failure partway through leaves the earlier
        batches written. That is tolerable here because ids are deterministic,
        so re-running overwrites rather than duplicating -- the ingestion is
        idempotent for a fixed chunk setting. Change the chunk setting and it is
        not; that needs ``reset()``.
        """
        if not chunks:
            return
        self._reject_oversized(chunks)
        size = self._batch_size()
        for start in range(0, len(chunks), size):
            batch = chunks[start : start + size]
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=self._embed([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "source": c.source,
                        "source_type": c.source_type,
                        "page": c.page,
                        "structural": c.structural,
                    }
                    for c in batch
                ],
            )

    def query(self, text: str, k: int = 5, include_structural: bool = False) -> list[Retrieved]:
        """Return the ``k`` chunks most similar to ``text``, best first.

        Reference lists, acknowledgments and contents pages are excluded **by
        default**, so no caller has to remember to ask for the filter and forgetting
        cannot quietly ship boilerplate to a student. ``include_structural=True``
        exists for the eval, which needs an unfiltered arm to measure the filter
        against, and for auditing what is being held back.

        Filtering happens in the store rather than after the fact because a
        post-filter would silently return fewer than ``k`` results -- on questions the
        corpus cannot answer, which is exactly where the boilerplate wins, that could
        be most of them.
        """
        where = None if include_structural else {"structural": False}
        res = self.collection.query(query_embeddings=self._embed([text]), n_results=k, where=where)
        return [
            Retrieved(
                text=doc,
                source=meta["source"],
                source_type=meta["source_type"],
                page=int(meta["page"]),
                score=1.0 - dist,  # cosine distance -> similarity
                structural=bool(meta.get("structural", False)),
            )
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def count(self, include_structural: bool = True) -> int:
        """Number of chunks indexed; by default every one, filtered or not.

        The default counts everything because that is what "the index holds N chunks"
        means to a reader of an eval report. Pass ``False`` for the retrievable
        subset -- the two together are what makes a report's provenance say which arm
        produced it.
        """
        if include_structural:
            return self.collection.count()
        return len(self.collection.get(where={"structural": False}, include=[])["ids"])

    def sources(self) -> set[str]:
        """Documents that actually have chunks in the index.

        Read from the index rather than from ``data/raw``, because those two can
        disagree -- an interrupted ingestion leaves a document on disk but not
        indexed. Provenance stamped on an eval report has to describe what was
        searched, not what was available.
        """
        got = self.collection.get(include=["metadatas"])
        return {str(m["source"]) for m in got["metadatas"]}

    def pages_present(self, source: str, include_structural: bool = False) -> set[int]:
        """Pages of ``source`` with at least one chunk that a query could return.

        Exists for the eval harness to check its own ground truth: a label
        naming a page that was never indexed -- a typo, or a page whose text
        would not extract -- can never be retrieved, so it scores as a miss that
        looks like poor retrieval. Same shape of bug as the silent truncation:
        the number comes out wrong and nothing complains.

        Structural chunks are excluded for that same reason, and it is the point of
        the default: a labelled page whose every chunk was filtered is just as
        unreachable as one that was never indexed, so it has to fail the same check
        rather than quietly become a miss.
        """
        where: dict = {"source": source}
        if not include_structural:
            where = {"$and": [where, {"structural": False}]}
        got = self.collection.get(where=where, include=["metadatas"])
        return {int(m["page"]) for m in got["metadatas"]}
