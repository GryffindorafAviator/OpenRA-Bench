"""Shared pytest collection guards for optional Rust-engine tests."""
from __future__ import annotations

import importlib.util
import inspect

import pytest


_ENGINE_SKIP = pytest.mark.skip(reason="Rust env wheel not installed")
_ENGINE_TOKENS = (
    "run_level(",
    "RustEnvPool(",
    "run_handoff(",
    "open_study_session(",
    "import openra_train",
    # Helper in tests/test_perception_ablation.py that boots a real
    # Rust env pool — the test bodies don't carry the explicit tokens
    # above, only this helper-call signature, so detect it directly.
    "_render_state(",
)


def pytest_collection_modifyitems(config, items):
    """Skip live Rust-engine tests when the openra_train wheel is absent.

    CI for this repo installs Python requirements but does not build the
    optional OpenRA-Rust extension. Structural/schema tests should still run;
    only tests that directly exercise the live engine via run_level or
    RustEnvPool are skipped.
    """
    if importlib.util.find_spec("openra_train") is not None:
        return
    for item in items:
        try:
            src = inspect.getsource(item.obj)
        except (OSError, TypeError):
            continue
        if any(token in src for token in _ENGINE_TOKENS):
            item.add_marker(_ENGINE_SKIP)
