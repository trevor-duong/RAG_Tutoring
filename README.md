# RAG Tutoring

A retrieval-augmented generation system grounded in my deep learning tutoring
materials. Students ask questions about DL concepts — backprop, loss functions,
CNNs/RNNs/Transformers — and get answers cited to real source passages rather
than generic LLM output.

Pilot target: real tutoring students, summer 2026.

## Status

**Phase 3 — frontend + deploy.** Source PDFs are chunked, embedded locally, stored
in Chroma, served over HTTP as cited passages, and searchable from a single page.
Deployed to Fly.io behind a shared password, index baked into the image. The full
corpus is indexed — 9,169 chunks from 37 documents — and retrieval quality is
measured rather than eyeballed:

```bash
python scripts/run_eval.py --save eval/baseline.json
```

Current baseline is **recall@5 = 0.50** over 32 labelled questions, with the
score broken out by how the question is phrased. See `eval/README.md` for what
the numbers do and do not support.

A single-page frontend at `/` consumes that API, behind a shared password. Next:
2–3 real students. Retrieval improvements are otherwise deferred and written up as
experiments to run against this baseline rather than done now — one has been run so
far, described below.

### Reference lists do not compete with explanations

8.5% of the corpus is a document's *apparatus* rather than its exposition:
bibliographies, acknowledgments, contents pages. It chunks and embeds like prose
and competes for the five slots a student sees. `structure.py` classifies it and
`VectorStore.query` excludes it by default.

The classification is per *chunk*, never per page, and that distinction is the
whole design. Page 14 of the Lottery Ticket paper is its acknowledgments page —
and it also holds four chunks of appendix prose, one of which is the top hit for
a real question about overfitting. Filtering the page would have deleted the
answer in order to remove the apparatus.

Chunks are flagged and filtered at query time rather than dropped at ingest, so
both arms of the comparison come from one index — `run_eval.py
--include-structural` reproduces the pre-filter numbers exactly, from the same
embeddings. On this corpus the filter moved recall@10 from 0.69 to 0.72 and MRR
from 0.319 to 0.323; the honest summary is that it fixed the top-5 for two
questions and rescued one from a total miss. `eval/README.md` names them.

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

### Turning generation on

Retrieval needs no key. To also get synthesised answers, copy `.env.example` to
`.env` and fill in `ANTHROPIC_API_KEY` (from
[console.anthropic.com](https://console.anthropic.com) — Console billing is
separate from a Claude.ai subscription). `.env` is gitignored and is read only by
the two entry points that can need a key: the API's lifespan and
`run_generation_eval.py`. A real environment variable beats the file, which is how
a deployment supplies the key — nothing there reads a file.

`GET /health` is the authoritative answer for whether it took: `generation_model`
is the model name, or `null` when generation is off.

## Reproducing the index and the numbers

Run in this order from an empty state. Each step's output is what the next one
reads, so a partial run is visible rather than silent.

```bash
python scripts/build_index.py     # ~14 min: extract, chunk, embed, index all documents
python scripts/audit_corpus.py    # verify nothing was silently corrupted or dropped
python scripts/audit_structure.py # verify the structural filter is not eating content
python scripts/run_eval.py --save eval/baseline.json
```

`build_index.py` resets and rebuilds rather than adding incrementally, so the
index only ever contains chunks from the current chunk setting — which is what
makes `eval/baseline.json` mean something. It also caches extracted page text to
`data/processed/pages.jsonl`, because extraction dominates the runtime; the audit
and `scripts/find_passage.py` read that cache instead of paying for it again.

For a change that alters chunking or chunk metadata but not extraction,
`build_index.py --from-cache` re-chunks that cache instead. What that skips is
extraction; what it still pays is embedding 9,169 chunks, which is most of the
cost — measured at 21 minutes on 2026-08-17. (The 71 seconds this used to claim
was measured on a much smaller corpus and never re-checked. A number in the docs
that nothing re-derives drifts silently, which is the same failure mode as an
eval baseline nobody re-runs.) It is only valid while `ingest.load_pdf` is unchanged, and the
guard is downstream rather than in the script: an unchanged setting has to
reproduce the saved baseline exactly, and a stale cache would not.

`find_passage.py` does a lexical search over the corpus and is how eval ground
truth gets located — never with the retriever, which would make the eval
circular:

```bash
python scripts/find_passage.py "internal covariate shift" --source "Batch Norm"
```

### Scoring generation

Separate instrument, separate baseline: generation writes to
`eval/generation-baseline.json`, so a prompt change cannot move a retrieval number.

`eval/baseline.json` tracks the production config rather than staying frozen, and
its `provenance` records which arm produced it — two reports with the same chunk
count can still describe different candidate sets. What makes overwriting it safe
is that the earlier arm is *reproducible* rather than merely preserved:
`run_eval.py --include-structural` regenerates the pre-filter numbers from the
same embeddings, which is a stronger guarantee than a stale file, since a file
cannot be re-derived after the index moves on.

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

Needs an index already built (the sequence above), and a password.

```bash
RAG_ACCESS_PASSWORD=... uvicorn rag_tutoring.api:app --reload
```

```bash
curl -s -u any:$RAG_ACCESS_PASSWORD -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why does my training loss go down but validation error go up?","k":3}'
```

`GET /health` reports what is actually indexed — model, chunk budget, collection,
chunk and document counts — because a result set means little without the index
that produced it. Interactive docs at `/docs`.

### The password is not optional

`/ask` returns verbatim source text and the corpus includes one purchased document, so
`RAG_ACCESS_PASSWORD` gates both `/ask` and the page at `/`. **Unset, the API refuses to
start** — it is not a feature that defaults to off.

That is the opposite of how the API key behaves, and the asymmetry is the point: with no
key, generation is off and passages still serve, which is a working product. With no
password, the corpus is served to whoever finds the URL. An optional feature may default
to off only when off is the safe state.

It is HTTP Basic with one shared password and no username (any username is accepted —
all the entropy is in the password). The browser prompts on navigation to `/` and then
attaches the same credentials to the page's own `fetch("/ask")`, which is why the
frontend contains no login form and no token. `/health` is deliberately open, so a
platform health check can reach it; it reports counts and model names, never document
text. Accounts, revocation and rate limiting are Phase 4 — see
`docs/decisions/0002-deploying-the-pilot.md`.

### Where the index lives

`RAG_CHROMA_DIR` overrides it; unset, it is `chroma/` in the checkout. That variable
is the only path configuration a deployment needs, and it exists because the served
app and the toolchain want different things from the filesystem.

The toolchain — ingestion, the audits, the eval — reads source PDFs, an extraction
cache and the eval set, all of which only exist in a checkout. The API reads the index
and nothing else. So `config.py` resolves the first group through functions that raise
when there is no repository, and the second from the environment. Before that split
every path was a module-level constant computed at import from a walk up out of the
working directory, which raised when it found no `pyproject.toml` — fine on a laptop,
and a startup crash in a container, where the package sits in site-packages and no
checkout exists. `api.py` imports `config`, so the failure landed before uvicorn could
bind a port.

The repository is located from `__file__` rather than from the working directory,
which matters in the one case that would otherwise hide the bug: an image built with
`COPY . /app` has a `pyproject.toml` above the working directory, so a cwd-based search
would succeed, quietly default the index to `/app/chroma`, and leave `RAG_CHROMA_DIR`
never exercised.

`tests/test_config.py` runs in a subprocess against a staged copy of the package,
because pytest always has the repo root available — an in-process test here would pass
with the fix reverted.

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

## Deploying

One image holds the API, the embedding weights and the index. The reasoning is in
`docs/decisions/0002-deploying-the-pilot.md`; the operational shape is:

```bash
docker build -t rag-tutoring .
docker run --rm --network none -p 8000:8000 -e RAG_ACCESS_PASSWORD=... rag-tutoring
```

`--network none` is not paranoia, it is the test. The embedding model is downloaded at
build time and `HF_HUB_OFFLINE=1` is set at runtime, so a container that can still answer
with no network is a container that is not quietly fetching 87 MB of weights from the
Hugging Face Hub the first time a student asks something. Check `/ask` rather than
`/health` — `/health` answers from state stamped at startup and would look fine over a
broken index, and Chroma opens `chroma.sqlite3` read-write even for pure reads, so a
permissions mistake on `/app/chroma` shows up only on a query.

Note what a local build on an Apple Silicon machine does and does not prove. It produces
a linux/arm64 image; Fly's builder produces linux/amd64, and that is the artifact that
gets deployed. The Dockerfile is written to be true on both — it asserts
`torch.version.cuda is None` rather than a `+cpu` version suffix, because that suffix
only exists on x86_64 and a suffix check would fail a perfectly correct aarch64 build.
`flyctl deploy` needs no local daemon at all, and `HF_HUB_OFFLINE=1` means a missing
model is a **boot failure** rather than a silent download, so a healthy `/health` on the
deployed machine carries most of what `--network none` was standing in for.

**The index is baked in, so re-indexing is a redeploy.** `COPY chroma/ /app/chroma` plus
`RAG_CHROMA_DIR=/app/chroma`. That gives the image and the index it serves one version
number, which is what makes `eval/baseline.json` mean anything about what students are
actually querying. The gate before building is that the eval reproduces the committed
baseline — after a deploy is too late, because the artifact is already in use.

It also means the deploy has to **upload local files** rather than build from a git
checkout: `chroma/` is gitignored, so a GitHub-connected build has no index to bake. Both
`flyctl deploy` and `railway up` upload the working directory and honour
`.dockerignore`; the git integration is the thing that cannot work here. Fly was chosen
from there.

```bash
flyctl secrets set RAG_ACCESS_PASSWORD=...   # required; the app will not start without it
flyctl secrets set ANTHROPIC_API_KEY=...     # optional; turns generation on
flyctl deploy
```

Secrets live only in the platform. Nothing in the repository or the image contains one,
and nothing in a deployment reads a file — `.env` is a local-development convenience
only.

**The image is as sensitive as `data/`.** It contains the chunked text of every source
document, including the purchased one. Private registry only.

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
Dockerfile           The deployable artifact: API + embedding weights + index.
```

Directories for later phases exist as placeholders. They stay empty until the
phase that fills them.

## Decisions

Choices with a real tradeoff behind them are written up in `docs/decisions/` —
context, the call, and the condition that should make me revisit it. This
doubles as system design interview prep: each record is something I should be
able to explain out loud.
