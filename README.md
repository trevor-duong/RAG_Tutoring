# RAG Tutoring

A retrieval-augmented generation system grounded in my deep learning tutoring
materials. Students ask questions about DL concepts — backprop, loss functions,
CNNs/RNNs/Transformers — and get answers cited to real source passages rather
than generic LLM output.

Pilot target: real tutoring students, summer 2026.

## Status

**Phase 1 complete; starting Phase 2 — API + eval.** Source PDFs are chunked,
embedded locally, stored in Chroma, and queried to return the most relevant
passages with citations (document title + page); the pipeline is walkable in
`notebooks/01_ingest_and_retrieve.ipynb`. The full corpus is indexed — 9,169
chunks from 37 documents — and retrieval quality is now measured rather than
eyeballed:

```bash
python scripts/run_eval.py --save eval/baseline.json
```

Current baseline is **recall@5 = 0.50** over 32 labelled questions, with the
score broken out by how the question is phrased. See `eval/README.md` for what
the numbers do and do not support. FastAPI and citation formatting are next.

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

Then drop source documents into `data/raw/papers/` and `data/raw/textbooks/`
(gitignored — the materials stay local).

## Reproducing the index and the numbers

Run in this order from an empty state. Each step's output is what the next one
reads, so a partial run is visible rather than silent.

```bash
python scripts/build_index.py     # ~14 min: extract, chunk, embed, index all documents
python scripts/audit_corpus.py    # verify nothing was silently corrupted or dropped
python scripts/run_eval.py --save eval/baseline.json
```

`build_index.py` resets and rebuilds rather than adding incrementally, so the
index only ever contains chunks from the current chunk setting — which is what
makes `eval/baseline.json` mean something. It also caches extracted page text to
`data/processed/pages.jsonl`, because extraction dominates the runtime; the audit
and `scripts/find_passage.py` read that cache instead of paying for it again.

`find_passage.py` does a lexical search over the corpus and is how eval ground
truth gets located — never with the retriever, which would make the eval
circular:

```bash
python scripts/find_passage.py "internal covariate shift" --source "Batch Norm"
```

To explore the pipeline interactively:

```bash
jupyter lab   # notebooks/01_ingest_and_retrieve.ipynb
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
scripts/             Rebuild the index, audit it, find passages, run the eval.
tests/               Unit tests for the logic that is easy to get subtly wrong.
data/raw/            Source tutoring materials (gitignored).
data/processed/      Extracted page text, cached between runs (gitignored).
eval/                Eval set: questions with known-correct sources, plus results.
docs/decisions/      Architecture decision records.
```

Directories for later phases exist as placeholders. They stay empty until the
phase that fills them.

## Decisions

Choices with a real tradeoff behind them are written up in `docs/decisions/` —
context, the call, and the condition that should make me revisit it. This
doubles as system design interview prep: each record is something I should be
able to explain out loud.
