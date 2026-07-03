from __future__ import annotations

from samosbor.config import read_secret_from_env_or_file


def test_read_secret_from_file_accepts_bearer_prefix(monkeypatch, tmp_path):
    monkeypatch.delenv("TBANK_INVEST_TOKEN", raising=False)
    token_file = tmp_path / "token.txt"
    token_file.write_text("Bearer abc.def_12345678901234567890", encoding="utf-8")

    token = read_secret_from_env_or_file(
        "TBANK_INVEST_TOKEN",
        str(token_file),
        label="test token",
    )

    assert token == "abc.def_12345678901234567890"


def test_read_secret_prefers_env_over_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TBANK_INVEST_TOKEN", "env.token_12345678901234567890")
    token_file = tmp_path / "token.txt"
    token_file.write_text("file.token_12345678901234567890", encoding="utf-8")

    token = read_secret_from_env_or_file(
        "TBANK_INVEST_TOKEN",
        str(token_file),
        label="test token",
    )

    assert token == "env.token_12345678901234567890"
