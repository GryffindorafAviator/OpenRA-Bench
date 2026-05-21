"""combat-target-priority-highvalue pack — full no-cheat validation.

Wave-11 ACTION pack: threat-weighted target prioritization (SC2
focus-fire target priority / military target prioritization anchor).
A 4-tank squad faces a mixed enemy cluster — a screen of cheap rifle
chaff (e1) backed by THREE high-threat anti-armour rocket soldiers
(e3). The squad must FOCUS THE ROCKET SOLDIERS FIRST: silence the
rocket soldiers fast, then mop up the chaff. Killing the chaff first
leaves all three rockets firing anti-armour fire through the entire
mop-up.

Win = units_killed_gte:K AND own_units_gte:3 AND within_ticks:2700
AND building_count_gte:fact:1 (a non-combat anchor: the squad must
still own its construction yard). Fail floors are aligned to the win
floor so every non-timeout outcome is a real WIN or LOSS — no
dead-zone DRAW.

Recalibrated after the engine movement fixes (moving units take fire
en route; attack_unit on out-of-sight targets paths normally; no
sprint-invincibility). Finding: with the post-fix combat model the
squad takes ~the same tank losses regardless of fine target order
(concentrating fire bunches the stack's return-fire exposure too) —
the survival floor is own_units_gte:3 (a perfect focus engagement
still loses ~1 tank closing the distance), NOT the old zero-loss
own_units_gte:4. The chaff-vs-threat discrimination is restored on
medium + hard via a BIGGER cluster (12 e1 + 3 e3 = 15, kill bar 15):
a chaff-first play cannot clear all 15 in budget while the three e3
attrit it, so it busts the kill bar AND the floor.

Bar (per CLAUDE.md), verified deterministic across seeds 1-4 on
every level:
  • stall (observe only)   → LOSS every level/seed (after_ticks).
  • brute attack_move      → LOSS every level/seed (drives into the
    cluster, bleeds 2 tanks, fails the kill bar).
  • kill-chaff-first       → LOSS on MEDIUM + HARD (the 15-strong
    cluster cannot be cleared chaff-first in budget; the e3 attrit
    the squad). EASY is the forgiving bare-skill tier — its smaller
    12-unit cluster lets a chaff-first play still finish, so the
    load-bearing chaff-first LOSS is medium + hard.
  • focus-threats-first    → WIN every level/seed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "combat-target-priority-highvalue.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)
# enemy cluster advance target — the chaff column, mid-y.
_ADV = (66, 20)


# ── helpers ───────────────────────────────────────────────────────


def _tank_ids(obs):
    return [
        str(u["id"])
        for u in (obs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]


def _enemy_units(obs):
    """(id, type) for every visible enemy combat unit."""
    out = []
    for e in (obs.get("enemy_summary") or []):
        t = str(e.get("type", "")).lower()
        if t in ("e1", "e3"):
            out.append((str(e.get("id")), t))
    return out


# ── policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — the kill bar is never met, after_ticks LOSS."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _brute_policy():
    """attack_move onto the cluster centroid — the squad drives INTO
    the cluster and is enveloped; it bleeds two tanks and fails to
    clear the kill bar before the deadline ⇒ LOSS."""
    def pol(obs, Cmd):
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        es = [
            e for e in (obs.get("enemy_summary") or [])
            if str(e.get("type", "")).lower() in ("e1", "e3")
        ]
        if es:
            cx = sum(e["cell_x"] for e in es) // len(es)
            cy = sum(e["cell_y"] for e in es) // len(es)
            return [Cmd.attack_move(ids, cx, cy)]
        return [Cmd.attack_move(ids, _ADV[0], _ADV[1])]
    return pol


def _focus_policy(first: str):
    """attack_unit, prioritising the `first` unit type. first='e1' is
    the kill-chaff-first trap; first='e3' is the intended threat-first
    focus play."""
    def pol(obs, Cmd):
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        es = _enemy_units(obs)
        prio = [e for e in es if e[1] == first]
        rest = [e for e in es if e[1] != first]
        if prio:
            return [Cmd.attack_unit(ids, prio[0][0])]
        if rest:
            return [Cmd.attack_unit(ids, rest[0][0])]
        # no enemy in view — close to contact range.
        return [Cmd.attack_move(ids, _ADV[0], _ADV[1])]
    return pol


# ── tests ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on the deadline — no draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE, got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_brute_attack_move_loses(level, seed):
    """A brute attack-move play auto-targets the chaff screen; the
    three rocket soldiers fire through the engagement and wipe the
    squad — a real LOSS, not a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"{level}/seed{seed}: brute must LOSE, got {res.outcome}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_kill_chaff_first_loses(level, seed):
    """Explicitly attacking the cheap e1 chaff first leaves the three
    rockets firing anti-armour fire through the whole mop-up; on the
    15-strong medium/hard cluster the squad cannot clear all 15 in
    budget and the e3 attrit it below the survival floor — a real
    LOSS. EASY is excluded: its smaller 12-unit cluster is the
    forgiving bare-skill tier where a chaff-first play can still
    finish (the load-bearing chaff-first LOSS is medium + hard)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _focus_policy("e1"), seed=seed)
    assert res.outcome == "loss", (
        f"{level}/seed{seed}: kill-chaff-first must LOSE, got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_focus_threats_first_wins(level, seed):
    """The intended capability — concentrate all four tanks on the
    rocket soldiers FIRST — must WIN every level and seed."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _focus_policy("e3"), seed=seed)
    assert res.outcome == "win", (
        f"{level}/seed{seed}: focus-threats-first must WIN, got {res.outcome}"
    )


def test_hard_agent_spawn_axis_has_two_groups():
    """The hard tier must define ≥2 agent-side spawn_point groups
    (the seed-driven staging-corridor axis); the strike force and its
    construction yard are duplicated under each group."""
    c = compile_level(load_pack(PACK), "hard")
    agent_sps = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert len(agent_sps) >= 2, (
        f"hard needs ≥2 agent spawn_point groups, got {sorted(agent_sps)}"
    )


def test_tick_budget_alignment():
    """within_ticks / after_ticks must be reachable inside max_turns
    (tick ≤ 93 + 90·(max_turns-1)) so the deadline actually bites."""
    for level in LEVELS:
        c = compile_level(load_pack(PACK), level)
        max_tick = 93 + 90 * (c.max_turns - 1)
        win_clauses = c.win_condition.all_of or []
        within = next(
            (cl["within_ticks"] for cl in win_clauses if "within_ticks" in cl),
            None,
        )
        assert within is not None and within <= max_tick, (
            f"{level}: within_ticks {within} not reachable by {max_tick}"
        )
        fail_clauses = c.fail_condition.any_of or []
        after = next(
            (cl["after_ticks"] for cl in fail_clauses if "after_ticks" in cl),
            None,
        )
        assert after is not None and after <= max_tick, (
            f"{level}: after_ticks {after} not reachable by {max_tick}"
        )
