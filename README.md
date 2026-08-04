# RAG Tutoring

A retrieval-augmented generation system grounded in my deep learning tutoring
materials. Students ask questions about DL concepts — backprop, loss functions,
CNNs/RNNs/Transformers — and get answers cited to real source passages rather
than generic LLM output.

Pilot target: real tutoring students, summer 2026.

## Status

**Phase 3 — frontend + deploy.** Source PDFs are chunked, embedded locally, stored
in Chroma, served over HTTP as cited passages, and searchable from a single page.
Not yet deployed. The full corpus is indexed —
9,169 chunks from 37 documents — and retrieval quality is measured rather than
eyeballed:

```bash
python scripts/run_eval.py --save eval/baseline.json
```

Current baseline is **recall@5 = 0.50** over 32 labelled questions, with the
score broken out by how the question is phrased. See `eval/README.md` for what
the numbers do and do not support.

A single-page frontend at `/` consumes that API. Next: deploy, then 2–3 real
students. Retrieval improvements are deliberately deferred and written up as
experiments to run against this frozen baseline, not done now.

### The API retrieves; it does not answer

`POST /ask` returns the passages that best match a question, each cited, and
leaves the reading to the student. There is no generation step, on purpose: a
correct page is in the top 5 half the time but is the *top* hit only 19% of the
time, so synthesising one confident answer from the top hits would turn a
retrieval miss into fluent prose a student cannot audit. Showing five sources
with their scores puts the judgement where the evidence supports it. Generation,
when it comes, consumes this same response.

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

## Running the API

Needs an index already built (the sequence above).

```bash
uvicorn rag_tutoring.api:app --reload
```

```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"Why does my training loss go down but validation error go up?","k":3}'
```

`GET /health` reports what is actually indexed — model, chunk budget, collection,
chunk and document counts — because a result set means little without the index
that produced it. Interactive docs at `/docs`.

Citations say **"PDF page 14"**, not "p. 14". The page is pypdf's index, which
matches a PDF viewer's counter but *not* the number printed on the page — front
matter puts those 18 apart in *Dive into Deep Learning*. A bare "p. 14" would be
followable and wrong; see `src/rag_tutoring/citations.py`.

## The frontend

`GET /` serves one self-contained HTML file — `src/rag_tutoring/static/index.html`,
inline CSS and JS, no build step. It is packaged data, so `pip install .` carries
it into a container; `unzip -l` on a built wheel is the check that it did.

One file served by the same process, rather than a separate app, means one
deployable artifact and no CORS surface at all, because the page and the API share
an origin. A component framework would buy routing and reuse that one page has no
use for.

The page fixes `k=5` and does not expose it. The baseline is recall@5, so five is
the number the measurement supports; a control inviting `k=20` would return a tail
that reads as more evidence and is mostly noise. `MAX_K` stays where it belongs, as
a server-side guard. Styling is lifted from
[trevor-duong.github.io](https://trevor-duong.github.io) so the two read as one
portfolio.

Two frontend rules are pinned by tests, because both fail silently in a browser
where no Python test would see them: the page renders the server's `page_label`
verbatim and never composes a reference from the raw index, and it references only
routes this app actually registers.

## Development

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
src/rag_tutoring/    Library code: ingest, store, evaluate, citations, api.
src/rag_tutoring/static/  The one page the API serves.
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
