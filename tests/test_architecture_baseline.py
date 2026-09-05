"""M0.5 deterministic guards for the architecture migration.

These assertions intentionally use only scenario loading/compilation, so they
run without the native ``openra_train`` wheel.  They pin two representative
active packs (perception and action) at the public boundary that M1 will wrap.

Engine, playback, and interactive-session baselines remain in their existing
specialised integration suites; see ``docs/ARCHITECTURE_MIGRATION_INVENTORY.md``.
"""

from __future__ import annotations

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level


@pytest.mark.parametrize(
    ("filename", "pack_id", "capability", "max_turns"),
    [
        # These use checked-in scalar map ids at easy level. Do not replace
        # them with procedural-map fixtures here: compilation of those packs
        # materializes generated .oramap files and would make this read-only
        # architectural guard mutate the worktree.
        ("scout-far-frontier.yaml", "scout-far-frontier", "perception", 17),
        ("strategy-twobody.yaml", "strategy-twobody", "action", 100),
    ],
)
def test_representative_scenarios_load_and_compile_deterministically(
    filename: str,
    pack_id: str,
    capability: str,
    max_turns: int,
) -> None:
    """The M1 scenario facade must preserve legacy loader/compiler meaning."""
    pack = load_pack(PACKS_DIR / filename)
    compiled = compile_level(pack, "easy")

    assert pack.meta.id == pack_id
    assert pack.meta.capability == capability
    assert compiled.pack_id == pack_id
    assert compiled.level == "easy"
    assert compiled.meta.capability == capability
    assert compiled.max_turns == max_turns
    assert compiled.map_supported is True
    assert compiled.fog_mode == "vision"
