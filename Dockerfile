# The deployable artifact: the API, its model, and the index it serves, in one image.
#
# Baking the index in is the decision worth explaining (docs/decisions/0002). A result
# set means little without the index that produced it -- `/health` says so, and
# `eval/baseline.json` depends on it -- so the image and the index it serves get one
# version number instead of two. The cost is that a re-index is a redeploy.
#
# Build and run from the repo root:
#   docker build -t rag-tutoring .
#   docker run --rm -p 8000:8000 -e RAG_ACCESS_PASSWORD=... rag-tutoring

FROM python:3.11-slim

# Duplicated from config.EMBEDDING_MODEL so the weights can be fetched before the app
# source is copied -- editing a module must not re-download 87 MB. The duplication is
# checked below rather than trusted: drift here would mean downloading one model at
# build time and asking for another at runtime, under HF_HUB_OFFLINE.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# HF_HOME is a real ENV, not a variable on the RUN below: set only at build time, the
# runtime would look in ~/.cache/huggingface, miss the pre-download, and re-fetch.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

# torch first, and from the CPU index, in its own layer. The default resolution pulls
# the CUDA build -- roughly 2 GB of kernels for hardware this will never have. The index
# carries only torch-family packages, so it cannot be used for the whole install; the
# assertion is what catches a silent fall-back to the default wheel.
#
# It checks `torch.version.cuda`, not a `+cpu` version suffix. The suffix only exists on
# x86_64, where there is a CUDA build to disambiguate from -- an aarch64 CPU wheel has no
# suffix, so a suffix check would fail a perfectly correct build. Fly runs x86_64; asking
# the question that is true on both is free.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && python -c "import torch; assert torch.version.cuda is None, f'CUDA build slipped in: {torch.__version__}'"

# The embedding weights, baked. Without this the first student question reaches the
# Hugging Face Hub from inside the container: slow, and a dependency on a third party
# being up to answer a question about a document that is already local.
RUN pip install --no-cache-dir "sentence-transformers>=3.0" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

# The application. Installed from /src and then deleted, deliberately: an installed copy
# in site-packages with no pyproject.toml above it is the shape config.py resolves paths
# for. `COPY . /app` would instead leave a pyproject.toml above the working directory,
# and a cwd-based search would have found it, silently defaulted the index to
# /app/chroma, and left RAG_CHROMA_DIR never exercised.
COPY pyproject.toml /src/
COPY src/ /src/src/
RUN pip install --no-cache-dir /src && rm -rf /src

# Three build-time assertions, because each of these fails silently at runtime.
RUN python -c "\
from rag_tutoring import config; \
assert config._REPO_ROOT is None, f'this image contains a checkout at {config._REPO_ROOT}'; \
assert config.EMBEDDING_MODEL == '$EMBEDDING_MODEL', f'model drift: {config.EMBEDDING_MODEL}'; \
from pathlib import Path; \
assert Path(config.__file__).parent.joinpath('static/index.html').exists(), 'the page did not ship'"

# Offline only from here: an accidental fetch has to fail loudly rather than work on a
# laptop and hang in production. Telemetry off for the same reason it is off everywhere
# else -- nothing about this corpus leaves the machine.
ENV HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False \
    TOKENIZERS_PARALLELISM=false \
    RAG_CHROMA_DIR=/app/chroma

RUN useradd --create-home --uid 10001 app

# Chroma opens chroma.sqlite3 read-write and creates -wal/-shm beside it even for pure
# reads, so the *directory* has to be writable by the running user. Owned by root, the
# image boots, /health answers, and only /ask fails.
COPY --chown=app:app chroma/ /app/chroma/
RUN chown -R app:app /opt/huggingface

USER app
WORKDIR /app
EXPOSE 8000

# PORT is honoured for platforms that inject it; Fly sets the port in fly.toml instead.
# `exec` so uvicorn is PID 1 and receives the platform's SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn rag_tutoring.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
