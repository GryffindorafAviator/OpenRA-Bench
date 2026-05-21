"""build-production-throughput-multibuilding pack — full no-cheat
validation on Rust.

REASONING — parallel manufacturing throughput. The agent starts with a
full base and ONE war factory (weap). A fixed quota of 6 medium tanks
(2tnk) must be fielded before a tight tick deadline. A SINGLE war
factory is a serial production line — it cannot clear the quota in
time. The agent has cash for a SECOND war factory; two factories
produce vehicles IN PARALLEL (the parallel-production engine fix —
OpenRA-Rust/openra-sim/src/world.rs), roughly doubling output, which
is the only play that hits the deadline.

Bar (CLAUDE.md "no defect, no cheat, no draw"):
  * stall (observe only) LOSES every (level, seed) — 0 tanks, the
    deadline bites.
  * one-weap-only (spam 2tnk from the single pre-placed factory,
    never build the 2nd) LOSES every (level, seed) — a serial
    factory clears only ~5 tanks by the deadline.
  * intended (queue the 2nd weap turn 1, place it, keep BOTH vehicle
    queues saturated) WINS every (level, seed).
  * Real LOSS not DRAW — `fail after_ticks:T+1` inside max_turns is
    the bite; a far unarmed enemy `fact` marker gates engine
    auto-done so a non-winner reaches the deadline.
  * hard tier defines ≥2 agent spawn_point groups (NORTH y=14 /
    SOUTH y=26 base) round-robined by seed.

Measured timing (installed wheel, seed 1, both queues saturated):
  ONE weap:  tank #5 fields ≈ tick 2613, tank #6 ≈ tick 3063.
  TWO weap:  tank #6 fields ≈ tick 2253, tank #7 ≈ tick 2613.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-production-throughput-multibuilding.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""

    def pol(obs, Cmd):
        return [Cmd.observe()]

    return pol


def _one_weap_policy():
    """Wrong play: spam 2tnk from the single pre-placed war factory and
    never build the second. A serial production line cannot clear the
    6-tank quota before the deadline — must LOSE every (level, seed)."""

    def pol(obs, Cmd):
        prod = obs.get("production", []) or []
        in_flight = sum(1 for p in prod if p == "2tnk")
        cmds = [Cmd.build("2tnk") for _ in range(max(0, 4 - in_flight))]
        return cmds or [Cmd.observe()]

    return pol


def _intended_policy():
    """Parallel-throughput play: queue the SECOND war factory on turn 1,
    place it adjacent to the existing weap (read from the live obs so
    the policy generalises across the hard-tier spawn variation), and
    keep BOTH vehicle queues saturated. Must WIN every (level, seed)."""
    placed = [False]

    def pol(obs, Cmd):
        cmds = []
        own_b = obs.get("own_buildings", []) or []
        types = [b["type"] for b in own_b]
        prod = obs.get("production", []) or []
        nweap = types.count("weap")
        weap = [b for b in own_b if b["type"] == "weap"]
        if nweap < 2 and "weap" not in prod and not placed[0]:
            cmds.append(Cmd.build("weap"))
        if nweap < 2 and "weap" in prod and weap:
            wx, wy = int(weap[0]["cell_x"]), int(weap[0]["cell_y"])
            cmds.append(Cmd.place_building("weap", wx + 3, wy))
            placed[0] = True
        in_flight = sum(1 for p in prod if p == "2tnk")
        for _ in range(max(0, 5 - in_flight)):
            cmds.append(Cmd.build("2tnk"))
        return cmds or [Cmd.observe()]

    return pol


# ── Tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE (no tanks ⇒ quota missed); "
        f"got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_one_weap_only_loses(level, seed):
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, _one_weap_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: spamming tanks from a single war factory "
        f"must LOSE (a serial line cannot clear the 6-tank quota in "
        f"time); got {res.outcome}. If this WINS, the parallel-"
        f"production capability is not load-bearing — re-tune N / T."
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_build_second_factory_wins(level, seed):
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, _intended_policy(), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed{seed}: building a SECOND war factory and feeding "
        f"both queues must WIN (parallel production doubles output); "
        f"got {res.outcome}"
    )
