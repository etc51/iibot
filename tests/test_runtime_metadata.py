from __future__ import annotations

from samosbor.runtime_metadata import current_commit_hash, with_runtime_metadata


def test_current_commit_hash_prefers_environment_override(monkeypatch):
    current_commit_hash.cache_clear()
    monkeypatch.setenv("SAMOSBOR_COMMIT_HASH", "abc123-runtime")

    assert current_commit_hash() == "abc123-runtime"

    current_commit_hash.cache_clear()


def test_with_runtime_metadata_preserves_existing_commit_hash():
    payload = with_runtime_metadata({"commit_hash": "existing", "value": 1})

    assert payload == {"commit_hash": "existing", "value": 1}
