"""rob-unexpected-enemy-spawn scenario family, full loop on Rust.

Wave-6 ROBUSTNESS / adversarial robustness. The pack tests handling
of a SURPRISE 2nd wave that closes in from a fog corner from a
different bearing than the obvious 4-unit Wave-1 cluster.

The win predicate makes BOTH waves load-bearing:

* `units_killed_gte:{7|8|9}` ⇒ the threat is actually handled (NOT
  just Wave 1 — the bar requires both); a forward-chase that runs
  out of force before both waves are dead can never satisfy this;
* `has_building:fact` ⇒ the base survives — chasing Wave 2 deep into
  the fog corner while Wave 1 walks past the unguarded base razes
  the fact;
* `own_units_gte:2` ⇒ a minimum reserve survived — attrition busts
  (forward-rush, stall) get the screen wiped (the whole garrison
  lost → fail);
* `within_ticks:5400` paired with `after_ticks:5401` ⇒ a non-finisher
  is a real reachable timeout LOSS (90 turns × ≤90 ticks/step
  reaches ≥5400 in interrupt mode), never a draw.

These tests prove with deterministic scripted policies (no model,
no network) that:

* the intended engage-but-don't-chase-deep policy WINS every level +
  every hard seed (1..4);
* stall / forward-rush both LOSE every level + every hard seed (a
  real LOSS, not a draw);
* the `after_ticks` deadline is reachable inside `max_turns`;
* the hard tier defines ≥2 spawn_point groups (so the fog corner
  the 2nd wave arrives from varies by seed — single-cell
  memorisation cannot generalise).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "rob-unexpected-enemy-spawn.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ─────────────────────────────────────────────────


def _own_ids(rs):
    return [str(u["id"]) for u in (rs.get("units_summary") or [])]


def _fact_xy(rs):
    fact = next(
        (b for b in (rs.get("own_buildings") or []) if b.get("type") == "fact"),
        None,
    )
    if fact is None:
        return None
    return int(fact["cell_x"]), int(fact["cell_y"])


def _visible_enemy_units(rs):
    """List of (x, y) for every VISIBLE enemy infantry (not the inert
    fact marker far east)."""
    raw = rs.get("_raw") or {}
    ep = raw.get("enemy_positions") if isinstance(raw, dict) else None
    out = []
    if isinstance(ep, list):
        for e in ep:
            if isinstance(e, dict) and e.get("actor_type") in ("e1", "e3"):
                out.append((int(e.get("cell_x", 0)), int(e.get("cell_y", 0))))
    return out


def stall(rs, C):
    """Observe-only baseline. The defenders are stance:2 (Defend) so
    they auto-fire on a hostile inside their arc but never advance.
    An un-commanded garrison meets the hunters piecemeal, splits its
    fire, and is swarmed and wiped — kills never reach the bar and
    own_units drops to zero → fail-clause fires (LOSS, not a draw)."""
    return [C.observe()]


def forward_rush(rs, C):
    """Defeat-first-only-relax taken to its degenerate limit: chase
    the visible Wave-1 cluster all the way to its centre x=80, y=20.
    The screen is pulled FAR east of the base; when Wave 2 INJECTS
    mid-episode at the NE fog corner (90, 8) at tick 1500, the screen
    has already wandered east of the base and is engaged in melee
    bombardment from Wave 1. The screen takes attrition and either
    gets wiped (`own_units_gte:1` fail) or fails to meet the kill
    bar before the `after_ticks:5401` deadline → LOSS on every level
    + seed."""
    own = _own_ids(rs)
    if not own:
        return [C.observe()]
    return [C.attack_move(own, 80, 20)]


def make_intended():
    """Engage the closest visible threat with `attack_move`, but CAP
    the chase distance so the defender screen never strays more than
    ~12 cells from the fact. When the 2nd wave closes in from the
    fog corner the screen is already pulled back to base-adjacent
    range and intercepts before fact contact. Both waves dead, fact
    alive, own_units_gte:2 satisfied → WIN every level + seed."""

    def policy(rs, C):
        own = _own_ids(rs)
        if not own:
            return [C.observe()]
        xy = _fact_xy(rs)
        if xy is None:
            return [C.attack_move(own, 30, 20)]
        fx, fy = xy
        threats = _visible_enemy_units(rs)
        if threats:
            threats.sort(key=lambda p: math.hypot(p[0] - fx, p[1] - fy))
            ex, ey = threats[0]
            # Reserve-capacity guardrail: don't chase enemies > 12
            # cells from the fact — fall back to a base-adjacent ring.
            if abs(ex - fx) > 12 or abs(ey - fy) > 12:
                return [C.attack_move(own, fx + 6, ey)]
            return [C.attack_move(own, ex, ey)]
        return [C.attack_move(own, fx + 4, fy)]

    return policy


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_hunt_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "rob-unexpected-enemy-spawn"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (the four named in meta).
    anchors = pack.meta.benchmark_anchor
    assert any("adversarial robustness" in a for a in anchors), anchors
    assert any("follow-on" in a for a in anchors), anchors
    assert any("SC2" in a for a in anchors), anchors
    # Hunt bot must be wired through to the engine for every level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "hunt", (lvl, bot)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns (≤90 ticks/step in
    interrupt mode)."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert deadline < reachable, (
        f"{level}: deadline {deadline} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the fog corner the surprise wave arrives from varies by seed
    (anti-memorisation)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126):
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_starting_cash_zero_for_all_levels():
    """Reserve-capacity discrimination: no cash + no tent = the model
    CANNOT train fresh defenders, so the binding decision is
    RESERVING the starting force, not out-producing the threat."""
    pack = load_pack(PACK)
    assert pack.starting_cash == 0
    # No barracks pre-placed on any level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_buildings = {
            a.type for a in c.scenario.actors
            if a.owner == "agent" and a.type in ("tent", "barr", "hand")
        }
        assert not agent_buildings, (lvl, agent_buildings)


def test_anti_draw_marker_present_on_every_level():
    """Anti-DRAW: an unarmed enemy `fact` marker must persist past
    the last hunter death so the engine doesn't auto-`done` on
    enemy-elim before the win/fail evaluator runs (CLAUDE.md rule 5)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        enemy_facts = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert len(enemy_facts) >= 1, lvl


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_engage_no_overcommit_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended engage-no-overcommit play "
            f"must WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / brute policy LOSES (not draws) ────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("forward_rush", lambda: forward_rush),
    ],
)
def test_lazy_and_overcommit_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (passive — bleeds out under attrition) and forward-rush
    (over-commits to the fog corner — base unguarded / screen wiped)
    must BOTH LOSE on every level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ───────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended(), seed=3)
    b = run_level(c, make_intended(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
