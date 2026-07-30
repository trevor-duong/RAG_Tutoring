# RAG Tutoring

A retrieval-augmented generation system grounded in my deep learning tutoring
materials. Students ask questions about DL concepts — backprop, loss functions,
CNNs/RNNs/Transformers — and get answers cited to real source passages rather
than generic LLM output.

Pilot target: real tutoring students, summer 2026.

## Status

**Phase 1 — ingestion + basic retrieval, notebook-level.** Working end to end
in `notebooks/01_ingest_and_retrieve.ipynb`: source PDFs are chunked, embedded
locally, stored in Chroma, and queried to return the most relevant passages
with citations (document title + page). Running against a starter slice of the
corpus; scaling to the full set and building the Phase 2 eval set are next.

## Roadmap

Phases are ordered deliberately and not skipped ahead of.

1. **Ingestion + retrieval.** Ingest 10–15 docs, get retrieval working in a
   notebook. No API.
2. **API + eval.** Wrap in FastAPI, add citation formatting, build a 20–30
   question eval set with known-correct sources.
3. **Frontend + deploy.** Minimal UI, deploy to Railway/Fly.io, test with 2–3
   real students.
4. **Infra**, in this order: caching → async ingestion (job queue) →
   observability/logging → rate limiting / vector DB scaling.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The editable install is what makes `from rag_tutoring import ...` work inside
notebooks without path hacks.

Then drop source documents into `data/raw/` (gitignored — the materials stay
local) and start JupyterLab:

```bash
jupyter lab
```

Lint, format, and test:

```bash
ruff check .
ruff format .
pytest
```

## Layout

```
src/rag_tutoring/    Library code. Imported by notebooks now, by the API later.
notebooks/           Phase 1 exploration.
tests/               Unit tests for the logic that is easy to get subtly wrong.
data/raw/            Source tutoring materials (gitignored).
data/processed/      Chunked/derived artifacts (gitignored).
eval/                Phase 2 eval set: questions with known-correct sources.
docs/decisions/      Architecture decision records.
```

Directories for later phases exist as placeholders. They stay empty until the
phase that fills them.

## Decisions

Choices with a real tradeoff behind them are written up in `docs/decisions/` —
context, the call, and the condition that should make me revisit it. This
doubles as system design interview prep: each record is something I should be
able to explain out loud.
