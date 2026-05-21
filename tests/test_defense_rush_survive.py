"""defense-rush-survive scenario family, full loop on Rust.

No-cheat redesign: the pack tests OPENING rush-defense while sustaining
baseline throughput against a relentless concentrated rusher band
(scripted bot `rusher`, charges the agent's centroid every ~8 ticks).
The win predicate makes BOTH axes load-bearing:

* `building_count_gte:{pbox,1}` and `building_total_gte:4` ⇒ at least
  one new defensive structure beyond the pre-placed fact+tent+powr —
  brute-army-of-e1s alone can never satisfy this;
* `own_units_gte:3` ⇒ the throughput SLA — all-defense (pillboxes only)
  can never satisfy this;
* `units_killed_gte:{4|6|8}` ⇒ the threat is actually handled, not
  ignored — all-economy can never satisfy this and gets the fact razed
  on the way;
* `within_ticks:4800` paired with `after_ticks:4801` ⇒ a non-finisher
  is a real reachable timeout LOSS (60 turns × ≤90 ticks/step reaches
  ≥4800 in interrupt mode), never a draw.

These tests prove with deterministic scripted policies (no model,
no network) that:

* the intended pbox+e1 defense-first policy WINS every level + every
  hard seed (1..4);
* stall / all-defense / all-economy / brute-army all LOSE every level +
  every hard seed (a real LOSS, not a draw);
* the `after_ticks` deadline is reachable inside `max_turns`;
* the hard tier defines ≥2 spawn_point groups (so the rush lane that
  matters varies by seed — single-cell memorisation cannot generalise).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "defense-rush-survive.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ─────────────────────────────────────────────────


def _fact_xy(rs):
    fact = next(
        (b for b in (rs.get("own_buildings") or [])
         if b.get("type") == "fact"),
        None,
    )
    if fact is None:
        return None
    return int(fact["cell_x"]), int(fact["cell_y"])


def stall(rs, C):
    """Observe-only — the agent never spends. Fact gets razed."""
    return [C.observe()]


def make_intended():
    """Defense-first intended policy: queue the pillbox on turn 1 and
    place it covering the lane mouth (Defense queue), THEN — once the
    pbox is standing and actively firing — keep training e1 defenders
    from the pre-placed barracks (Infantry queue). Prioritising the
    Defense queue gets the M60mg pillbox online before the concentrated
    rusher band arrives; the trained e1 defenders then satisfy the
    throughput SLA (own_units_gte) while the pbox holds the choke.
    Both axes (pbox + units) stay load-bearing — the e1 stream is still
    required for own_units_gte:3 and the pbox for the building bars."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        types = [b.get("type") for b in own_b]
        prod = rs.get("production") or []
        prod_items = [
            p.get("item") for p in prod if isinstance(p, dict)
        ]
        xy = _fact_xy(rs)
        if xy is None:
            return [C.observe()]
        fx, fy = xy
        cmds = []
        has_pbox = "pbox" in types
        # Defense queue FIRST: queue + place a pbox covering the lane
        # mouth without splitting cash against the Infantry queue.
        if not has_pbox:
            if "pbox" not in prod_items:
                cmds.append(C.build("pbox"))
            cmds.append(C.place_building("pbox", fx + 6, fy))
        # Infantry queue: once the pbox is up, keep training e1
        # defenders to satisfy the throughput SLA.
        units = rs.get("units_summary") or []
        n_units = sum(
            1 for u in units
            if str(u.get("type", "")).lower() in ("e1", "e3")
        )
        if has_pbox and n_units < 8 and "e1" not in prod_items:
            cmds.append(C.build("e1"))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_all_defense():
    """All-defense: only ever build pillboxes (no units, no economy).
    Satisfies the pbox/building bars but NEVER own_units_gte:3 →
    LOSS (and typically the fact dies anyway since the pbox line
    alone cannot blunt a concentrated rusher charge)."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        pbox_count = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [
            p.get("item") for p in prod if isinstance(p, dict)
        ]
        xy = _fact_xy(rs)
        if xy is None:
            return [C.observe()]
        fx, fy = xy
        if pbox_count < 3:
            cmds = []
            if "pbox" not in prod_items:
                cmds.append(C.build("pbox"))
            cmds.append(
                C.place_building("pbox", fx + 4 + pbox_count * 2, fy)
            )
            return cmds
        return [C.observe()]

    return policy


def all_economy(rs, C):
    """All-economy: only stand up extra power plants, never any
    defence or unit. The rusher reaches the fact and razes it →
    LOSS (and never has_building:pbox)."""
    own_b = rs.get("own_buildings") or []
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    xy = _fact_xy(rs)
    if xy is None:
        return [C.observe()]
    fx, fy = xy
    n_powr = sum(1 for b in own_b if b.get("type") == "powr")
    cmds = []
    if n_powr < 5:
        if "powr" not in prod_items:
            cmds.append(C.build("powr"))
        cmds.append(
            C.place_building("powr", fx + 2, fy + 4 + n_powr * 2)
        )
    if not cmds:
        cmds.append(C.observe())
    return cmds


def brute_army(rs, C):
    """Brute-army: only ever train e1, never a pillbox. Often kills
    enough rushers but never satisfies building_count_gte:pbox →
    a reachable timeout LOSS at the after_ticks deadline."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
        return [C.build("e1")]
    return [C.observe()]


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "defense-rush-survive"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors.
    anchors = pack.meta.benchmark_anchor
    assert any("MicroRTS" in a for a in anchors), anchors
    assert any("SC2LE" in a for a in anchors), anchors
    # Rusher bot must be wired through to the engine for every level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert (str(bot).lower() == "rusher"), (lvl, bot)


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
    so the lane that matters varies by seed (anti-memorisation)."""
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


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_balanced_policy_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended pbox+e1 defense-first play "
            f"must WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / single-axis policy LOSES (not draws) ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("all_defense", make_all_defense),
        ("all_economy", lambda: all_economy),
        ("brute_army", lambda: brute_army),
    ],
)
def test_lazy_and_single_axis_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (fact razed), all-defense (own_units_gte unmet OR fact
    razed), all-economy (no pbox / fact razed), and brute-army (no
    pbox at the deadline) must ALL LOSE on every level + every seed —
    no draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── attrition cap (hard) actually bites ───────────────────────────────


def test_hard_attrition_cap_is_enforced():
    """The hard `units_lost_lte:3` cap must be present in win AND fail
    so over-attrition cannot win and also explicitly loses."""
    c = compile_level(load_pack(PACK), "hard")
    win = c.win_condition.model_dump(exclude_none=True)
    fail = c.fail_condition.model_dump(exclude_none=True)
    win_cap = next(
        (clause["units_lost_lte"] for clause in win.get("all_of", [])
         if "units_lost_lte" in clause),
        None,
    )
    assert win_cap == 3, win
    has_fail_cap = any(
        (clause.get("not") or {}).get("units_lost_lte") == 3
        for clause in fail.get("any_of", []) or []
    )
    assert has_fail_cap, fail


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
