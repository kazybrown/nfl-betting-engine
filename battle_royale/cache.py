"""Disk cache for slate-level artifacts (reference field, dup index, sims).

Live drafts need answers in seconds, but the reference field, duplication
index and outcome-sim matrix each take seconds-to-minutes to build. They only
depend on the slate and the model configuration, so they are computed once
per slate and cached on disk, keyed by content hashes — a new rankings CSV or
any config/table change produces a new key automatically.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "battle_royale"


def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def text_hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()[:16]


def load(cache_dir: str | Path, key: str):
    path = Path(cache_dir) / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save(cache_dir: str | Path, key: str, obj) -> None:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{key}.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(d / f"{key}.pkl")
