"""Dependency-free .env loader: KEY=VALUE → os.environ, never
overriding an already-set var, quote-stripping, comment-skipping.
(The real .env holding the OpenRouter key is git-ignored and never
read in tests — we point the loader at a tmp file.)"""

from __future__ import annotations

import os

from openra_bench.run_eval import _load_dotenv


def test_loads_keys_skips_comments_and_strips_quotes(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        "OPENROUTER_API_KEY=sk-or-test-123\n"
        'QUOTED="value with spaces"\n'
        "BARE=plain\n"
        "MALFORMED_NO_EQUALS\n"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    _load_dotenv(f)
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-test-123"
    assert os.environ["QUOTED"] == "value with spaces"
    assert os.environ["BARE"] == "plain"
    assert "MALFORMED_NO_EQUALS" not in os.environ


def test_never_overrides_existing_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("OPENROUTER_API_KEY=from-file\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    _load_dotenv(f)
    assert os.environ["OPENROUTER_API_KEY"] == "from-shell"


def test_missing_file_is_a_noop(tmp_path):
    _load_dotenv(tmp_path / "nope.env")  # must not raise
