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
CHROMA_DIR = ROOT / "chroma"

# Embedding model. Changing this is a full re-index, not an incremental
# migration -- embeddings from two models are not comparable. See
# docs/decisions/0001-vector-store-and-embeddings.md.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking, in words. all-MiniLM-L6-v2 truncates input at 256 word-pieces
# (~1.3x words), so a ~160-word target keeps whole chunks inside the model's
# window instead of silently dropping their tails.
CHUNK_WORDS = 160
CHUNK_OVERLAP_WORDS = 40
