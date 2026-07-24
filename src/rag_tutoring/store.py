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

from rag_tutoring.config import CHROMA_DIR, EMBEDDING_MODEL
from rag_tutoring.ingest import Chunk


@dataclass(frozen=True)
class Retrieved:
    """One search hit: the chunk text, where it came from, and how close it was."""

    text: str
    source: str
    source_type: str
    page: int
    score: float  # cosine similarity in [0, 1]; higher is more relevant


class VectorStore:
    """Embeds chunks locally and stores them in a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        model_name: str = EMBEDDING_MODEL,
        collection: str | None = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
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

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks.

        Upsert (not add) so re-running an ingestion cell overwrites by id
        instead of erroring on duplicates.
        """
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=self._embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "source_type": c.source_type, "page": c.page} for c in chunks
            ],
        )

    def query(self, text: str, k: int = 5) -> list[Retrieved]:
        """Return the ``k`` chunks most similar to ``text``, best first."""
        res = self.collection.query(query_embeddings=self._embed([text]), n_results=k)
        return [
            Retrieved(
                text=doc,
                source=meta["source"],
                source_type=meta["source_type"],
                page=int(meta["page"]),
                score=1.0 - dist,  # cosine distance -> similarity
            )
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
            )
        ]

    def count(self) -> int:
        """Number of chunks currently indexed."""
        return self.collection.count()
