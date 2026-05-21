"""combat-target-priority-highvalue pack — full no-cheat validation.

Wave-11 ACTION pack: threat-weighted target prioritization (SC2
focus-fire target priority / military target prioritization anchor).
A 4-tank squad faces a mixed enemy cluster — a screen of cheap rifle
chaff (e1) backed by THREE high-threat anti-armour rocket soldiers
(e3). The squad must FOCUS THE ROCKET SOLDIERS FIRST: 4-vs-1
concentrated fire deletes each e3 in 1-2 decision turns before it can
land more than ~1 rocket. Killing the chaff first leaves all three
rockets firing through the entire mop-up and they whittle the squad
below the survival floor.

Win = units_killed_gte:K AND own_units_gte:F AND within_ticks:2700
AND building_count_gte:fact:1 (a non-combat anchor: the squad must
still own its construction yard while it focus-fires). Fail floors
are aligned to the win floor so every non-timeout outcome is a real
WIN or LOSS — no dead-zone DRAW.

Bar (per CLAUDE.md), verified deterministic across seeds 1-4 on
every level:
  • stall (observe only)   → LOSS every level/seed (after_ticks).
  • brute attack_move      → LOSS (auto-targets the chaff screen;
    the three e3s wipe the squad).
  • kill-chaff-first       → LOSS (rockets fire through the chaff
    mop-up and bust the survival floor).
  • focus-threats-first    → WIN every level/seed (all three e3s
    silenced inside the first ~3 turns, squad preserved).
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
    """attack_move onto the cluster centroid — the engine auto-targets
    the NEAREST hostile (the chaff screen), so the three e3s fire
    through the whole engagement and wipe the squad."""
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


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_kill_chaff_first_loses(level, seed):
    """Explicitly attacking the cheap e1 chaff first leaves the three
    rockets firing through the whole mop-up; the squad drops below the
    survival floor — a real LOSS."""
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
