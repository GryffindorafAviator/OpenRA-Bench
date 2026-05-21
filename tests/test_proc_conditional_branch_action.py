"""proc-conditional-branch-action pack — full no-cheat validation.

Wave-11 REASONING pack: conditional IF/ELSE procedure execution.
The brief is an explicit runbook fork — "scout the map; IF the
enemy outpost is on the NORTH flank, attack NORTH; ELSE attack
SOUTH". The agent must (1) SCOUT to observe which flank the
outpost is on, (2) BRANCH on that observation, (3) EXECUTE the
matching action (drive the strike force to the live flank and
raze the outpost).

The win is an explicit two-branch `any_of` IF/ELSE: each branch
is a `then`-chain `[buildings_discovered_gte:1,
enemy_key_buildings_destroyed_in_region{tent, flank}]` gated on a
region-scoped raze. The enemy outpost `tent` exists on only ONE
flank per seed (the Wave-9 enemy-side `spawn_point` axis), so
exactly one branch is satisfiable per seed.

Bar (per CLAUDE.md):
  • stall                → LOSS every level/seed (after_ticks).
  • always-attack-NORTH  → LOSS on the seeds where the outpost is
    SOUTH (hard seeds 1, 3); WINS only the NORTH seeds.
  • always-attack-SOUTH  → mirror: LOSS on the NORTH seeds.
  • scout-then-branch    → WIN every level + seed.
A fixed unconditional opener wins exactly one flank; only the
scout-then-branch policy generalises across the per-seed flip.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "proc-conditional-branch-action.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)
# On hard the env round-robins the enemy spawn_point per seed:
# seeds 1/3 → spawn_point 1 (outpost SOUTH), seeds 2/4 → spawn_point
# 0 (outpost NORTH). easy/medium have a single fixed NORTH outpost.
HARD_NORTH_SEEDS = (2, 4)
HARD_SOUTH_SEEDS = (1, 3)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on after_ticks every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _fixed_flank_policy(flank: str):
    """Always send the whole force to `flank` without scouting — a
    fixed unconditional branch. Correct on the seeds where the
    outpost is on `flank`, a stranded-army timeout LOSS otherwise."""
    outpost_y = 4 if flank == "north" else 36

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        cmds = [
            Cmd.attack_move([str(u["id"])], 20, outpost_y) for u in units
        ]
        return cmds or [Cmd.observe()]

    return pol


def _scout_then_branch_policy():
    """The intended capability: scout both flanks, branch on the
    observed enemy, then commit the strike force to the live flank."""
    state = {"flank": None}

    def _observed_flank(obs):
        ys = [
            e["cell_y"]
            for e in (obs.get("enemy_summary") or [])
            if e.get("cell_y") is not None
        ]
        ys += [
            b["cell_y"]
            for b in (obs.get("enemy_buildings_summary") or [])
            if b.get("cell_y") is not None
        ]
        if not ys:
            return None
        return "north" if sum(ys) / len(ys) < 20 else "south"

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        tanks = [u for u in units if u.get("type") == "2tnk"]
        flank = _observed_flank(obs)
        if flank and state["flank"] is None:
            state["flank"] = flank
        # Phase 1 — SCOUT: split the jeeps to probe both flanks.
        if state["flank"] is None:
            cmds = [
                Cmd.move_units([str(j["id"])], 20, 6 if i == 0 else 34)
                for i, j in enumerate(jeeps)
            ]
            return cmds or [Cmd.observe()]
        # Phase 2/3 — BRANCH + EXECUTE: drive the force to the live
        # flank and raze the outpost.
        outpost_y = 4 if state["flank"] == "north" else 36
        cmds = [
            Cmd.attack_move([str(u["id"])], 20, outpost_y)
            for u in tanks + jeeps
        ]
        return cmds or [Cmd.observe()]

    return pol


# ── Tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """A do-nothing policy must lose on the deadline — no draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE, got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_scout_then_branch_wins(level, seed):
    """The intended scout-then-branch policy must win every seed."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _scout_then_branch_policy(), seed=seed)
    assert res.outcome == "win", (
        f"{level}/seed{seed}: scout-then-branch must WIN, got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_flank_loses_on_opposite_seed_hard(seed):
    """On hard the outpost flank rotates per seed. A fixed
    unconditional opener must LOSE on the seeds where the outpost
    is on the other flank (and may win the matching seeds) — a
    2-of-4 play is a structural LOSS under the hard-tier contract."""
    c = compile_level(load_pack(PACK), "hard")
    north = run_level(c, _fixed_flank_policy("north"), seed=seed)
    south = run_level(c, _fixed_flank_policy("south"), seed=seed)
    if seed in HARD_NORTH_SEEDS:
        assert north.outcome == "win", (
            f"hard/seed{seed}: outpost is NORTH — always-NORTH should win"
        )
        assert south.outcome == "loss", (
            f"hard/seed{seed}: outpost is NORTH — always-SOUTH must LOSE"
        )
    else:  # HARD_SOUTH_SEEDS
        assert south.outcome == "win", (
            f"hard/seed{seed}: outpost is SOUTH — always-SOUTH should win"
        )
        assert north.outcome == "loss", (
            f"hard/seed{seed}: outpost is SOUTH — always-NORTH must LOSE"
        )


def test_hard_enemy_spawn_axis_has_two_groups():
    """The hard tier must define ≥2 enemy-side spawn_point groups
    (the Wave-9 per-owner seed axis) — the agent base stays fixed."""
    c = compile_level(load_pack(PACK), "hard")
    enemy_sps = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is not None
    }
    assert len(enemy_sps) >= 2, (
        f"hard needs ≥2 enemy spawn_point groups, got {sorted(enemy_sps)}"
    )
    agent_sps = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert not agent_sps, "agent base must be fixed (no agent spawn_point)"
