"""Cache / output locations for the competitor-analysis tool.

Cache goes to ``data/competitors/`` (sibling to ``data/journal`` etc., under the
repo root, git-ignored by ``data/``); the HTML report + intermediate analysis
go to ``model-lab/out/competitors/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import REPO_ROOT, _default_out_dir


def competitors_dir() -> Path:
    """``data/competitors`` cache dir (override: ``MODEL_LAB_COMPETITORS_DIR``)."""
    env = os.environ.get("MODEL_LAB_COMPETITORS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "data" / "competitors").resolve()


def out_dir() -> Path:
    """``model-lab/out/competitors`` report dir (respects ``MODEL_LAB_OUT``)."""
    base = os.environ.get("MODEL_LAB_OUT")
    root = Path(base).expanduser().resolve() if base else _default_out_dir()
    return (root / "competitors").resolve()


def account_dir(address: str) -> Path:
    """Per-account cache subdir, ``data/competitors/<address>``."""
    return competitors_dir() / address.lower()
