# 0002 — Deploying the pilot: where the index lives, and who gets in

Date: 2026-08-17
Status: Accepted (revisit at Phase 4)

## Context

Phase 3 ends with 2–3 real students using this from their own devices, which means
something publicly addressable for the first time. Three coupled questions had to be
settled before anything could be reachable.

`POST /ask` returns **verbatim source text**. The corpus is open papers and openly
licensed textbooks plus one purchased document carrying a per-buyer watermark. A URL
anyone can reach is therefore a redistribution question, not only a cost question.

The index is 89 MB of Chroma built locally from 806 MB of source PDFs -- 21 minutes of
embedding alone, on top of an extraction pass that is slower still. `data/` and `chroma/` are both gitignored, so nothing in the repository is
sufficient to reconstruct what the API serves.

And `eval/baseline.json` only means something in reference to a specific index —
`/health` reports chunk counts, collection and embedding model for exactly that reason.
A deployment that serves *some* index makes the committed numbers unfalsifiable.

## Decision

**A single image containing the API, the embedding weights and the index, deployed to
Fly.io with `flyctl deploy`, behind HTTP Basic with one shared password.**

- `COPY chroma/ /app/chroma`, and `RAG_CHROMA_DIR` points at it.
- `sentence-transformers/all-MiniLM-L6-v2` is downloaded at build time; `HF_HUB_OFFLINE=1`
  at runtime.
- `RAG_ACCESS_PASSWORD` gates `/ask` and `/`. Missing secret is a startup failure, not a
  mode. `/health` is exempt.
- Secrets — the access password and `ANTHROPIC_API_KEY` — are set with
  `flyctl secrets set` and exist nowhere in the repository or the image.

## Why

**The index is part of the artifact, not part of the environment.** Baking it in gives
the image and the index one version number, so "which index answered this?" has the same
answer as "which image is running?". Mounting a volume splits that into two independent
versions and adds a failure mode with no error message: an image serving an index nobody
can identify, still reporting counts through `/health` as though they were provenance.
The price is that re-indexing is a redeploy, which for a corpus that changes a few times
a phase is not a real cost.

**The platform constraint is upload-vs-git-checkout, not Fly-vs-Railway.** `chroma/` is
gitignored, so a build triggered by a git push has no index to bake — the artifact and
the deploy trigger disagree about what the source of truth is. `flyctl deploy` and
`railway up` both upload the local directory instead and honour `.dockerignore`, so both
work; a GitHub-connected build is the thing that does not. Fly was chosen on its own
merits from there.

**A shared password matches the size of the problem.** Accounts, sessions, per-user rate
limits and an audit trail are Phase 4 and are not what stands between a watermarked PDF
and the open internet — a password is. HTTP Basic specifically, because the browser
prompts on navigation and then attaches the same credentials to the page's own `fetch`,
so the frontend needs no login form and no token in JavaScript.

**Auth fails closed while generation fails open, and the asymmetry is deliberate.**
Generation with no key serves passages alone, which is a working product. Auth with no
secret serves the corpus to anyone, so there is no such state: the process refuses to
start. An optional feature may default to off only when off is the safe state.

**Offline at runtime is a correctness property, not an optimisation.** Loading the
embedding model from the Hub on first use means a student's first question depends on a
third party being reachable, and it works on a laptop while failing in a container.
Baking the weights and setting `HF_HUB_OFFLINE=1` converts that from a latent runtime
dependency into a build-time one that fails loudly.

## Revisit when

- **The corpus changes often enough that redeploying to re-index is the bottleneck.**
  That is the trigger for a volume, or for the vector store moving out of the process
  entirely — which is also ADR 0001's Phase 3 trigger.
- **More than a handful of students, or any need to know who asked what.** A shared
  password cannot be revoked per person and identifies nobody. Rotation is a redeploy of
  one secret today, which stops being acceptable the moment revocation has to be
  targeted.
- **The image outgrows the platform's build or boot limits**, at which point the index
  gets its own storage rather than the image getting smaller.
- **Any traffic that is not 2–3 known students.** The `k` bound in `AskRequest` is the
  hook rate limiting attaches to, and rate limiting is the Phase 4 item this defers.

## Consequences

- Re-indexing requires a redeploy. `build_index.py` → `run_eval.py` reproducing
  `eval/baseline.json` → `flyctl deploy` is now one sequence, not three independent ones.
- The image is as sensitive as `data/`: it contains the chunked text of the purchased
  document. It must never go to a public registry, which is a property of the deploy
  process and not something the code can enforce.
- A rebuilt index that is not evaluated is a silent regression. The gate is that the
  eval reproduces the committed baseline before the image is built — running it after a
  deploy is too late, because the artifact is already what students are using.
- The access password is shared and cannot be revoked for one person. If it leaks, it is
  rotated for everybody.
- `/health` is unauthenticated. It discloses counts, model names and a collection name,
  and no document text. That is a deliberate trade for a platform health check that
  cannot carry credentials.
