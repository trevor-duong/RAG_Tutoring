"""Tests for resolving paths, which is the difference between booting and not.

``config.py`` used to compute every path at import from a walk up out of the working
directory, and raise when it found no ``pyproject.toml``. That is fine on a laptop and
fatal in a container: ``pip install .`` puts the package in site-packages, nothing
above the working directory is a checkout, and ``import rag_tutoring.config`` raised
before uvicorn ever bound a port. ``api.py`` imports it, so the API could not start.

**These tests mostly run in a subprocess, and that is the whole point.** pytest runs
with the repo root right there -- findable from the working directory and from
``__file__`` alike -- so no in-process test can reproduce "there is no checkout". An
in-process test that set the environment variable and asserted it was honoured would
pass with the entire fix reverted, which would make it a test of nothing.

What the subprocess gets instead is a *copy of the package* in a temporary directory
with no ``pyproject.toml`` above it -- the shape of an installed wheel. It is copied
under a different name deliberately: the editable install registers ``rag_tutoring``
on this interpreter, and a same-named copy could resolve back to the checkout and pass
for the wrong reason. Nothing in ``config.py`` imports its own package, so the rename
costs nothing and the mechanism under test -- where ``config.py`` finds itself on
disk -- is exactly the one that runs in a container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from rag_tutoring import config

PACKAGE = Path(config.__file__).parent
STAGED = "deployed_pkg"


def _stage_installed_copy(tmp_path: Path) -> Path:
    """Copy the package somewhere with no checkout above it, and return that somewhere.

    ``tmp_path`` is under the OS temp directory, not under the repo, so the walk up
    from the copied ``config.py`` finds no ``pyproject.toml`` -- which is the condition
    a container is in and a developer's machine never is.
    """
    site = tmp_path / "site"
    shutil.copytree(PACKAGE, site / STAGED)
    assert not any((p / "pyproject.toml").exists() for p in (site / STAGED).parents), (
        "the staged copy must not sit under a checkout, or it tests nothing"
    )
    return site


def _run(site: Path, code: str, env_extra: dict[str, str] | None = None) -> str:
    """Run ``code`` against the staged copy, returning stdout. Raises on failure."""
    env = {k: v for k, v in os.environ.items() if k != config.CHROMA_DIR_ENV}
    env["PYTHONPATH"] = str(site)
    env.update(env_extra or {})
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=site.parent,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"subprocess failed:\n{done.stderr}"
    return done.stdout.strip()


def test_importing_config_does_not_require_a_repository(tmp_path):
    """The deploy blocker itself: importing must not need a checkout to exist."""
    site = _stage_installed_copy(tmp_path)
    where = _run(
        site,
        f"import {STAGED}.config as c; print(c.__file__); print(c._REPO_ROOT)",
    )
    loaded, root = where.splitlines()
    # If the editable install won the import, this is the checkout's copy and the
    # test below would be measuring the wrong file.
    assert str(tmp_path) in loaded, f"imported the wrong copy: {loaded}"
    assert root == "None", "a staged copy must not believe it is inside a checkout"


def test_without_a_checkout_the_index_comes_from_the_environment(tmp_path):
    site = _stage_installed_copy(tmp_path)
    elsewhere = tmp_path / "mnt" / "index"
    got = _run(
        site,
        f"import {STAGED}.config as c; print(c.chroma_dir())",
        {config.CHROMA_DIR_ENV: str(elsewhere)},
    )
    assert got == str(elsewhere)


def test_the_unset_index_error_names_the_variable_to_set(tmp_path):
    """The message is the only instruction a person reading container logs gets.

    The old failure said ``no pyproject.toml above cwd``, which is incomprehensible
    from inside a container and names nothing that could be done about it.
    """
    site = _stage_installed_copy(tmp_path)
    message = _run(
        site,
        f"import {STAGED}.config as c\n"
        "try:\n"
        "    c.chroma_dir()\n"
        "except RuntimeError as exc:\n"
        "    print(exc)\n"
        "else:\n"
        "    raise SystemExit('chroma_dir() should have raised')\n",
    )
    assert config.CHROMA_DIR_ENV in message


def test_toolchain_paths_fail_only_when_asked_for(tmp_path):
    """Import succeeds; the source documents are what is unavailable, and say so."""
    site = _stage_installed_copy(tmp_path)
    out = _run(
        site,
        f"import {STAGED}.config as c\n"
        "for fn in (c.repo_root, c.data_raw_dir, c.pages_cache):\n"
        "    try:\n"
        "        fn()\n"
        "    except RuntimeError:\n"
        "        print(fn.__name__, 'raised')\n"
        "    else:\n"
        "        raise SystemExit(fn.__name__ + ' should have raised')\n"
        "print('env file:', c.local_env_file())\n",
    )
    assert "repo_root raised" in out
    assert "data_raw_dir raised" in out
    assert "pages_cache raised" in out
    # None, not a path: callers must skip loading rather than pass it to load_dotenv,
    # which treats None as "search upward for any .env" rather than as "do not".
    assert "env file: None" in out


def test_the_checkout_is_found_from_the_file_not_the_working_directory(tmp_path):
    """Running from anywhere must still resolve the repo -- this is the notebook case.

    The old walk-up started at the working directory, so this held only because
    Jupyter and uvicorn happened to be started somewhere under the repo.
    """
    env = {k: v for k, v in os.environ.items() if k != config.CHROMA_DIR_ENV}
    done = subprocess.run(
        [sys.executable, "-c", "from rag_tutoring import config; print(config.repo_root())"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(config.repo_root())


def test_the_environment_wins_over_the_checkout(monkeypatch):
    """A deployment that mounts the index elsewhere must not be overruled by a repo."""
    monkeypatch.setenv(config.CHROMA_DIR_ENV, "/mnt/index")
    assert config.chroma_dir() == Path("/mnt/index")


def test_the_checkout_is_the_default_when_the_environment_is_silent(monkeypatch):
    monkeypatch.delenv(config.CHROMA_DIR_ENV, raising=False)
    assert config.chroma_dir() == config.repo_root() / "chroma"


def test_the_index_location_is_read_per_call_not_captured_at_import(monkeypatch):
    """Two calls under two environments must disagree.

    This is what ``VectorStore(persist_dir=None)`` depends on. A default argument
    would have frozen the location at import of the first module to touch this one,
    so a deployment setting the variable later would be ignored with no error.
    """
    monkeypatch.setenv(config.CHROMA_DIR_ENV, "/mnt/one")
    first = config.chroma_dir()
    monkeypatch.setenv(config.CHROMA_DIR_ENV, "/mnt/two")
    assert first != config.chroma_dir()


def test_toolchain_paths_sit_where_the_repository_says():
    root = config.repo_root()
    assert config.data_raw_dir() == root / "data" / "raw"
    assert config.pages_cache() == root / "data" / "processed" / "pages.jsonl"
    assert config.local_env_file() == root / ".env"
