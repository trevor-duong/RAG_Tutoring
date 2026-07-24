# 0001 — Vector store and embedding model for Phase 1

Date: 2026-07-23
Status: Accepted (revisit at Phase 3)

## Context

Two coupled choices were left open in `CLAUDE.md`: pgvector vs Chroma for the
vector store, and sentence-transformers vs a hosted embedding API for computing
embeddings. Phase 1 is notebook-level retrieval over 10–15 documents — no HTTP
API, no deploy, no concurrent users, no persistence requirements beyond "don't
re-embed on every kernel restart."

Worth noting for the embedding half: Anthropic does not serve an embeddings
endpoint, so "embedding API" means taking on a second vendor purely for
embeddings.

## Decision

**Chroma** (embedded, persisting to a local directory) with
**sentence-transformers** running locally.

## Why

**Zero infrastructure.** Chroma persists to a directory. pgvector means running
Postgres — a container, a connection string, a schema, an extension — before a
single retrieval query has been written. That work is real, it just belongs to
Phase 3, when there's something deployed that needs it.

**No API key and no per-query cost.** The core activity of Phase 1 is tuning
chunking, and every chunking change means re-embedding the whole corpus. Local
embeddings make that loop free and offline, so the iteration count isn't
throttled by a bill or a rate limit.

**The corpus is small.** 10–15 documents is on the order of thousands of chunks.
Brute-force cosine similarity over that is milliseconds. Approximate nearest
neighbour indexing solves a problem this project does not have yet.

## Revisit when

Any one of these makes the decision worth reopening:

- **The API serves concurrent users.** An embedded, in-process store is the
  wrong shape once more than one worker needs the same index.
- **Postgres is already running for another reason** — users, eval results, the
  Phase 4 job queue. At that point one datastore beats two, and pgvector gets a
  lot more attractive than it is today.
- **Metadata filtering and vector search need to happen in one query**, or the
  corpus grows past roughly 100k chunks.
- **Retrieval quality plateaus** and the bottleneck is traced to the embedding
  model rather than to chunking. Swapping in a stronger hosted model is a
  contained change if the interface below holds.

## Consequences

Retrieval gets written against a narrow interface — add documents, query by
embedding, return chunks with metadata and scores. Holding that line means
changing stores touches one module, not the notebooks and not the future API
layer. If store-specific calls leak into calling code, this decision stops being
cheap to reverse, which is the main thing to protect.
