"""S9 spatial tensor reaches bench consumers end-to-end: pack →
eval_core → adapter.render_state carries a correctly-shaped occupancy
grid (the multimodal / ERQA-transfer substrate)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "perception-frontier-reading.yaml"


def test_spatial_tensor_flows_to_render_state():
    seen = {}

    def capture(render_state, Command):
        if "rs" not in seen:
            seen["rs"] = render_state
        return [Command.observe()]

    c = compile_level(load_pack(PACK), "easy")
    run_level(c, capture, seed=1)

    rs = seen["rs"]
    sp = rs["spatial"]
    h, w, ch = rs["spatial_shape"]
    assert ch == 6, f"6 channels, got {ch}"
    assert w > 0 and h > 0
    assert len(sp) == h * w * ch, "flat tensor length == h*w*c"

    def at(x, y, k):
        return sp[(y * w + x) * ch + k]

    # Fog channel (1): some cells visible (1.0) around starting units.
    assert any(sp[i] == 1.0 for i in range(1, len(sp), 6)), "fog has visible cells"
    # Passable channel (0): a non-trivial fraction of the map.
    passable = sum(sp[i] for i in range(0, len(sp), 6))
    assert passable > h * w * 0.1, "passable channel sane"
    # Own-unit channel (2): the agent's scout units are placed.
    assert any(sp[i] >= 1.0 for i in range(2, len(sp), 6)), "own units marked"
    # Indexing helper consistency.
    assert at(0, 0, 0) in (0.0, 1.0)


def test_spatial_absent_is_empty_not_error():
    """A render_state must always carry the keys (empty when the engine
    doesn't emit a tensor) so consumers can rely on them."""
    from openra_bench.rust_adapter import RustObsAdapter

    a = RustObsAdapter()
    a.observe({"unit_positions": {}, "game_tick": 1})  # no 'spatial'
    rs = a.render_state()
    assert rs["spatial"] == []
    assert rs["spatial_shape"] == (0, 0, 0)
