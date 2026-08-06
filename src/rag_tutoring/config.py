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

# Temperature 0 so a stored eval result is reproducible. Sampling would make a
# baseline that shifts under re-runs, and then a prompt change and noise would be
# indistinguishable -- the same reason the index is rebuilt rather than appended.
GENERATION_TEMPERATURE = 0.0

# Answers orient a student toward passages shown directly beneath them; they are not
# the deliverable on their own, so this is deliberately tight.
GENERATION_MAX_TOKENS = 700
