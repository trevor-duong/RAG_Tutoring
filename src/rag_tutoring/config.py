"""Central configuration: filesystem paths and the few tunable knobs.

Anything a person tuning the pipeline might want to change lives here, so the
notebooks and modules never hard-code a path or a magic number.

Paths come in two kinds, and the difference is what makes this module importable
in a container:

*Runtime* paths are what the served application reads -- in practice just the
index. They come from the environment first, so a deployment can put the index
anywhere without editing code.

*Toolchain* paths are the source PDFs, the extraction cache, the eval set: files
that only exist in a checkout, used by ingestion, the audits and the eval. They
are resolved relative to the repo root, which means they cannot be resolved at
all from an installed wheel -- so they are **functions, not constants**. A
function is where a failure that only some callers can provoke belongs. As
module-level constants they ran at import time and took the whole package down
with them, API included, on a machine that has no checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# The environment variable a deployment sets to say where the index lives. Named
# rather than inlined because the error raised when it is needed and missing quotes it,
# and that message is the only instruction a person reading container logs gets.
CHROMA_DIR_ENV = "RAG_CHROMA_DIR"


def _find_repo_root() -> Path | None:
    """Locate the checkout containing this file, or ``None`` if there isn't one.

    The root is the directory containing ``pyproject.toml``. Walking up from
    ``__file__`` rather than from the working directory is deliberate on both ends:
    under an editable install this file sits in ``src/rag_tutoring/``, so the root is
    found whether Jupyter or uvicorn was started from the repo root, from
    ``notebooks/``, or from anywhere else -- and under a real install into
    ``site-packages`` there is no ``pyproject.toml`` above it, so this returns ``None``
    and the container case is decided by where the code *is* rather than by what
    directory a process happened to start in.

    That second half is not a detail. A ``COPY . /app`` image with ``WORKDIR /app`` has
    a ``pyproject.toml`` above the working directory, so a cwd-based search would
    succeed there, silently default the index to ``/app/chroma``, and leave
    ``RAG_CHROMA_DIR`` never exercised -- the mechanism would look wired up while doing
    nothing.

    Returns ``None`` instead of raising because *this runs at import*. Every caller
    below that genuinely needs a checkout raises on its own behalf, with a message
    naming what it wanted.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return None


# Resolved once: a checkout does not move under a running process. Private, so that
# "is there a repo here" cannot be read as "here is a path" by a caller that then
# builds one out of it.
_REPO_ROOT = _find_repo_root()


def repo_root() -> Path:
    """The checkout root. Raises when the code is installed rather than checked out."""
    if _REPO_ROOT is None:
        raise RuntimeError(
            "no repository checkout found: this is an installed copy of rag_tutoring, "
            "so the source documents, the extraction cache and the eval set are not "
            "available. Only the API needs no checkout."
        )
    return _REPO_ROOT


def chroma_dir() -> Path:
    """Where the Chroma index lives -- the one path the served application needs.

    Read from the environment on every call rather than captured at import, so that a
    test can point it somewhere else and a process cannot be stuck with whatever the
    environment held at the moment the first module imported this one.
    """
    override = os.environ.get(CHROMA_DIR_ENV)
    if override:
        return Path(override)
    if _REPO_ROOT is None:
        raise RuntimeError(
            f"no index location: {CHROMA_DIR_ENV} is not set and this is an installed "
            f"copy of rag_tutoring with no repository to fall back on. Set "
            f"{CHROMA_DIR_ENV} to the directory holding the Chroma index."
        )
    return _REPO_ROOT / "chroma"


def data_raw_dir() -> Path:
    """Source tutoring materials. Gitignored -- the documents stay local.

    Named ``_dir`` to match ``chroma_dir`` and, more usefully, so it does not collide
    with ``ingest.corpus_documents``'s ``data_raw`` parameter: importing it under an
    alias to dodge the shadow made isort split the import into two blocks.
    """
    return repo_root() / "data" / "raw"


def pages_cache() -> Path:
    """Extracted page text, written by scripts/build_index.py during the indexing pass.

    PDF extraction is by far the slowest step (one 709 MB illustrated textbook takes
    ~10 minutes on its own), so the text is kept so that auditing the corpus and
    locating eval ground truth do not each pay for it again. Gitignored with the rest
    of ``data/``.
    """
    return repo_root() / "data" / "processed" / "pages.jsonl"


def local_env_file() -> Path | None:
    """A developer's local ``.env``, or ``None`` when there is no checkout to hold one.

    Read by the *entry points* (the API's lifespan and scripts/run_generation_eval.py)
    and by nothing else. Deliberately not loaded here at import time: the suite's
    defining property is that it runs with no key, and an import side effect would
    quietly hand pytest whatever key happens to be on the developer's disk. Gitignored.

    ``None`` means "do not look", and callers must honour it with a branch rather than
    hand it on: dotenv reads a ``None`` path as *search upwards for any ``.env`` you
    can find*, not as *skip*, which in a deployment is a wider blast radius than the
    explicit path it replaced. In a deployment nothing reads a file at all -- the
    platform injects real environment variables, and the application only ever reads
    os.environ.
    """
    return None if _REPO_ROOT is None else _REPO_ROOT / ".env"


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
