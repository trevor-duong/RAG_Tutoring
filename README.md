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

### The answer never replaces the sources

`POST /ask` returns a short answer *and* the passages it was written from, and the
page renders the passages underneath it. That shape is the design, and it comes
out of the baseline: a correct page is in the top 5 half the time but is the *top*
hit only 19% of the time, so an answer synthesised from these passages is
sometimes synthesised from the wrong ones. Returning the sources is what makes
that visible to a student rather than hidden behind fluent prose — answer-only is
the one variant the evidence does not support.

Two rules are enforced rather than hoped for:

- **Citations are validated, not just requested.** The model cites `[source N]`
  into the passage set it was handed, and every marker is checked against that set
  before the response leaves the server. If one doesn't resolve, the answer is
  dropped and the passages serve alone. Prompting for citations and trusting the
  output would be a check that cannot fail.
- **The model does not abstain, and does not invent.** Thin passages produce an
  answer that says what they do and don't cover plus the closest thing they
  genuinely say. "Always answer" must not decay into "answer from pretraining" — a
  hedged sentence sourced from the model's own memory is a confabulation wearing a
  disclaimer, and a student can't tell the difference.

Generation is **optional and off by default**. With no `ANTHROPIC_API_KEY` the API
serves passages alone, which is a working product rather than a degraded one; it
is exactly what this served before generation existed. That is also what keeps the
whole test suite runnable with no index, no embedding model, and no key.

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

### Scoring generation

Separate instrument, separate baseline. `eval/baseline.json` stays frozen so
retrieval changes remain comparable to every earlier run; generation writes to
`eval/generation-baseline.json`.

```bash
python scripts/run_generation_eval.py \
  --save eval/generation-baseline.json \
  --answers eval/generation-answers.jsonl
```

It scores the two partitions that carry the risk, and neither needs new labelling
because the existing labels already define them:

- **The ~16 misses.** recall@5 = 0.50, so about half the questions hand the model
  five passages containing no correct page. What it does there is the whole risk.
- **The 5 negatives.** The corpus can't answer these. A similarity threshold
  provably can't separate them from answerable questions — the score ranges
  overlap — but a model *reading* five passages is a different mechanism, so it can
  succeed where a scalar cutoff couldn't.

What the report measures is mechanical: does the answer cite, what does it cite,
does every citation resolve. Whether an answer is *faithful* needs a judge or a
human, so the ~21 misses and negatives get dumped for reading — at that size,
reading them is more honest than a proxy. `--save` writes metadata only; answer
text goes to the gitignored `--answers` file, because an answer may quote its
passages and `eval/baseline.json` carries no chunk text for the same reason.

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

Three frontend rules are pinned by tests, because all three fail silently in a
browser where no Python test would see them: the page renders the server's
`page_label` verbatim and never composes a reference from the raw index; it
references only routes this app actually registers; and it agrees with the server on
the `[source N]` citation marker, so a format change on one side can't turn every
citation into literal text sitting in the prose.

That marker format was measured, not picked. Bare `[N]` appears 1,342 times across
11.6% of the corpus's pages — papers cite by number — so validating a bare marker
would collide with the corpus's own references. `[source N]` appears zero times.

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
src/rag_tutoring/    Library code: ingest, store, evaluate, citations, generate, api.
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
