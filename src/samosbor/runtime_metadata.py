from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Mapping, MutableMapping

UNKNOWN_COMMIT_HASH = "unknown"


@lru_cache(maxsize=1)
def current_commit_hash() -> str:
    env_value = os.environ.get("SAMOSBOR_COMMIT_HASH", "").strip()
    if env_value:
        return env_value

    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return UNKNOWN_COMMIT_HASH

    return result.stdout.strip() or UNKNOWN_COMMIT_HASH


def with_runtime_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    enriched = dict(payload)
    enriched.setdefault("commit_hash", current_commit_hash())
    return enriched


def add_runtime_metadata(payload: MutableMapping[str, object]) -> MutableMapping[str, object]:
    payload.setdefault("commit_hash", current_commit_hash())
    return payload
