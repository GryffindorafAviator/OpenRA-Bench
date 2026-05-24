"""No-cheat bar guardrail for `spec-tanya-c4-strike`.

The pack tests the C4 demolition capability: agent moves Tanya to an
enemy refinery and fires `Command.c4_detonate`, which instantly
destroys the building. The win predicate is
`enemy_buildings_destroyed_gte: 1` plus a `within_ticks` deadline.

This test pins the SCENARIO-LEVEL no-cheat bar:

- STALL (`Command.observe()` only) must LOSE on every level and every
  hard seed. Historical defect: when Tanya was declared at the
  engine-default stance (Defend), she auto-fired on the nearby enemy
  refinery and her pistol destroyed the proc within the first decision
  tick — a pure stall policy thereby WON for free. The fix is
  `stance: 0` (HoldFire) on Tanya so the model must issue an EXPLICIT
  c4_detonate (or attack_unit) for the proc to fall.
- INTENDED (`c4_detonate(tanya, proc)`) must WIN on every level and
  every hard seed. This is the capability under test.

The brute `attack_unit` policy is not pinned here — Tanya's pistol
trivially destroys the proc at range, a pack-design issue that is
orthogonal to the stall defect this test guards against.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level


PACK_PATH = PACKS_DIR / "spec-tanya-c4-strike.yaml"


def _stall(_rs, C):
    return [C.observe()]


def _intended_c4(rs, C):
    """Issue c4_detonate(tanya, proc) when both are resolvable; the
    engine walks Tanya to the target and instantly destroys it."""
    units = rs.get("units_summary") or []
    tanya = next(
        (u for u in units if str(u.get("type", "")).lower() == "tanya"),
        None,
    )
    eb = rs.get("enemy_buildings_summary") or []
    proc = next(
        (b for b in eb if str(b.get("type", "")).lower() == "proc"),
        None,
    )
    if tanya and proc:
        return [C.c4_detonate([str(tanya["id"])], str(proc["id"]))]
    return [C.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level: str, seed: int) -> None:
    from openra_train import Command

    if not hasattr(Command, "c4_detonate"):
        pytest.skip(
            "Command.c4_detonate not available — rebuild the wheel "
            "(maturin develop --release) after the Tanya-C4 engine commit."
        )

    pack = load_pack(PACK_PATH)
    compiled = compile_level(pack, level)
    result = run_level(compiled, _stall, seed=seed)
    assert result.outcome == "loss", (
        f"stall policy must LOSE on {level} seed={seed}, "
        f"got outcome={result.outcome!r} turns={result.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_c4_wins(level: str, seed: int) -> None:
    from openra_train import Command

    if not hasattr(Command, "c4_detonate"):
        pytest.skip(
            "Command.c4_detonate not available — rebuild the wheel "
            "(maturin develop --release) after the Tanya-C4 engine commit."
        )

    pack = load_pack(PACK_PATH)
    compiled = compile_level(pack, level)
    result = run_level(compiled, _intended_c4, seed=seed)
    assert result.outcome == "win", (
        f"intended c4_detonate policy must WIN on {level} seed={seed}, "
        f"got outcome={result.outcome!r} turns={result.turns}"
    )
