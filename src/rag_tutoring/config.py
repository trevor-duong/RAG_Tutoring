"""Central configuration: filesystem paths and the few tunable knobs.

Anything a person tuning the pipeline might want to change lives here, so the
notebooks and modules never hard-code a path or a magic number.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Walk up from the current working directory to the repo root.

    The root is the directory containing ``pyproject.toml``. This lets a
    notebook resolve paths whether Jupyter was launched from the repo root or
    from ``notebooks/``.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate project root (no pyproject.toml above cwd)")


ROOT = project_root()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "chroma"

# Local-development secrets, read by the *entry points* (the API's lifespan and
# scripts/run_generation_eval.py) and by nothing else. Deliberately not loaded here at
# import time: the suite's defining property is that it runs with no key, and an import
# side effect would quietly hand pytest whatever key happens to be on the developer's
# disk. Gitignored. In a deployment nothing reads this file -- the platform injects real
# environment variables, and the application only ever reads os.environ.
ENV_FILE = ROOT / ".env"

# Extracted page text, written by scripts/build_index.py during the indexing
# pass. PDF extraction is by far the slowest step (one 709 MB illustrated
# textbook takes ~10 minutes on its own), so the text is kept so that auditing
# the corpus and locating eval ground truth do not each pay for it again.
# Gitignored with the rest of data/ -- the source materials stay local.
PAGES_CACHE = DATA_PROCESSED / "pages.jsonl"

# Embedding model. Changing this is a full re-index, not an incremental
# migration -- embeddings from two models are not comparable. See
# docs/decisions/0001-vector-store-and-embeddings.md.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Hard limit of the embedding model: input longer than this is truncated, and
# sentence-transformers does it *silently*. Chunks are therefore measured in
# word-pieces, not words -- a word costs anywhere from 1 to ~4 word-pieces in
# this corpus (math notation and long technical terms split hard), so no word
# count can guarantee a chunk fits. Measured: a 160-word chunk ranged from 90
# to 549 word-pieces, and 34% of them overflowed.
MODEL_MAX_TOKENS = 256

# Chunking budget, in word-pieces. Sits under MODEL_MAX_TOKENS with room for
# the [CLS]/[SEP] the tokenizer adds, plus margin.
CHUNK_MAX_TOKENS = 240
CHUNK_OVERLAP_TOKENS = 60

# Generation model, used to synthesise an answer from retrieved passages. Changing
# this changes what `eval/generation-baseline.json` is comparable to -- the same rule
# as EMBEDDING_MODEL, for the same reason. Unlike the embedding model it forces no
# re-index: generation reads the passages retrieval already found, so a model swap is
# a re-run of the generation eval, not a rebuild.
GENERATION_MODEL = "claude-sonnet-5"

# None means "send no sampling parameter at all", and is not a placeholder for a value
# to be filled in later. claude-sonnet-5 rejects both `temperature` and `top_p` with a
# 400 -- verified against the live API, not assumed -- so there is no sampling control
# to set on the configured model.
#
# This cost a property the eval was designed around. The intent was temperature 0, so a
# stored baseline would not shift under re-runs and a prompt change could be told apart
# from noise. That is no longer available here, which means
# `eval/generation-baseline.json` is a **sample, not a constant**: a small movement
# between runs is decoding variance, and only a large one is evidence of anything. Kept
# as a knob rather than deleted because an older model (claude-sonnet-4-6, checked) still
# accepts it, and because a deleted parameter reads as an oversight a later change would
# quietly restore.
GENERATION_TEMPERATURE: float | None = None

# Not an answer-length setting. claude-sonnet-5 returns `thinking` blocks by default and
# they are billed against this same ceiling, while `generate.py` keeps only the `text`
# blocks -- so this budget is shared with reasoning a student never sees. At 700 the
# 2026-08-05 eval truncated 2 of 37 answers mid-word, one of them mid-qualification.
#
# Answer length is held down by the prompt ("keep it short"), which is the right place
# for it: a prompt that asks for brevity produces a complete short answer, whereas a
# tight ceiling produces a long answer with its ending cut off. Raising this costs
# nothing -- output tokens are billed as generated, not as reserved -- so the ceiling
# exists only to bound a runaway.
GENERATION_MAX_TOKENS = 2000
