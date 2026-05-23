"""build-defensive-tower-line scenario family, full loop on Rust.

The pack tests DEFENSIVE PERIMETER TOPOLOGY ACROSS A WIDE FRONT: when
the threat is a rush spread across the FULL VERTICAL WIDTH of the map
(multiple distinct rows simultaneously), not pinched through a single
corridor cell, the right architecture is one pbox per row across the
FULL width (a LINE). A dense cluster on one row leaves every other row
unguarded; a scatter near the base never engages the rush. This is the
sibling/inverse of `def-tower-line-vs-cluster` (which forces a CLUSTER
at a single bottleneck cell); together the two packs discriminate
whether the model understands the FORCING GEOMETRY (single-cell
chokepoint vs wide-front approach).

Anchors: ERQA spatial commit / MicroRTS defense placement / military
perimeter (firewall rule placement).

The pbox is the load-bearing weapon. After the engine pbox-weapon fix
(`fix(engine): pbox gets a direct-fire Armament`) a BUILT pbox is an
active direct-fire anti-infantry tower. The rush arrives as a
`scheduled_events: spawn_actors` wave (or two waves, on hard) EAST of
the central column spread across multiple distinct rows AFTER the
agent has had time to build its LINE serially, and the `rusher` bot
charges the agent fact on the west, so each row's spawn group walks
WEST through the x=60 column on its starting y. There are NO pre-
placed agent defenders, so the pbox LINE is the sole source of kill
output.

The win predicate makes the LINE topology load-bearing — total pbox
count alone is not enough:

* `building_count_gte:{pbox, n:K}` ⇒ the agent built the full budget
  (K = 3 easy / 5 medium / 6 hard);
* `building_in_region:{pbox, x:60, y:Y, radius:0.5, count:1}` for
  EACH of the K front rungs ⇒ exactly one pbox per row across the
  front (a tiny radius 0.5 means only the exact cell counts, so a
  cluster on (60,20) misses all flank rungs and a scatter near the
  base misses every rung);
* `units_killed_gte:K` ⇒ the pbox LINE must actively KILL the rush
  spread across the front (a stall / pure-army layout kills 0);
* `building_count_gte:{fact,n:1}` (present-tense — `has_building` is
  the one-shot "ever-seen" set, see CLAUDE.md footgun);
* `within_ticks` paired with `after_ticks` in the fail clause ⇒ a
  non-finisher is a real reachable timeout LOSS (no interrupts on this
  pack ⇒ each step is exactly 90 ticks, so max_turns is a hard tick
  budget that the `after_ticks` deadline reliably bites in).

Per-tier design:
* easy   — 3-pbox LINE (rungs y=8/20/32), budget $1800, one wave.
* medium — 5-pbox LINE (rungs y=4/12/20/28/36), budget $3000, one wave.
* hard   — 6-pbox LINE (rungs y=4/10/16/22/28/34), budget $4800
  (= 6 rungs + 2 rebuilds), TWO scheduled waves (tick 1800 + tick
  3000) so a rung the first wave razes must be REBUILT before wave 2.
  Hard also flips the agent base latitude per seed (NORTH y=12 /
  SOUTH y=28 via spawn_point) so a memorised relative-to-base
  placement cannot generalise.

The scripted-policy validations prove deterministically that:

* the intended LINE policy (one pbox at each front rung, with rebuild
  on hard) WINS every level + every hard seed (1..4);
* stall / cluster-on-centre / scatter-near-base all LOSE every level +
  every hard seed — a real LOSS, not a draw (the rung clauses are
  never satisfied);
* the hard tier defines ≥2 spawn_point groups (NORTH base y=12 / SOUTH
  base y=28) so a memorised base-relative placement cannot generalise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-defensive-tower-line.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-tier rung topology (front rows across the full vertical width at
# x=60). A cluster on the centre row misses every flank rung because the
# region predicate uses radius 0.5 (cell-exact).
RUNGS_BY_LEVEL = {
    "easy":   [(60, 8),  (60, 20), (60, 32)],
    "medium": [(60, 4),  (60, 12), (60, 20), (60, 28), (60, 36)],
    "hard":   [(60, 4),  (60, 10), (60, 16), (60, 22), (60, 28), (60, 34)],
}

CASH_BY_LEVEL = {"easy": 1800, "medium": 3000, "hard": 4800}

# Cells used by the "scatter near base" wrong-topology policy: pboxes
# clustered near the base rather than across the front. None of these
# lie inside ANY rung region (radius 0.5 around the rung cells), so the
# region clauses are all unsatisfied.
SCATTER_NEAR_BASE_BY_LEVEL = {
    "easy":   [(20, 18), (22, 20), (24, 22)],
    "medium": [(20, 18), (22, 20), (24, 22), (26, 19), (18, 21)],
    "hard":   [(20, 18), (22, 20), (24, 22), (26, 19), (18, 21), (16, 23)],
}

# Cells used by the "cluster on centre row" wrong-topology policy:
# pboxes piled on the centre rung (y=20). Satisfies the count clause
# (and on hard meets the y=20-adjacent rungs if they exist) but misses
# every other rung because radius 0.5 is cell-exact.
CLUSTER_ON_CENTRE_BY_LEVEL = {
    "easy":   [(60, 19), (60, 20), (60, 21)],
    "medium": [(60, 18), (60, 19), (60, 20), (60, 21), (60, 22)],
    "hard":   [(60, 17), (60, 18), (60, 19), (60, 20), (60, 21), (60, 22)],
}


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. Fact gets razed by the
    rush AND the count/region clauses are never satisfied."""
    return [C.observe()]


def make_line(level: str):
    """Intended LINE topology: one pbox at EACH front rung. On hard the
    policy also REBUILDS any rung the first wave razes (the cash budget
    has slack for ≤2 rebuilds across the two-wave attrition)."""
    rungs = RUNGS_BY_LEVEL[level]

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        pboxes = [b for b in own_b if b.get("type") == "pbox"]
        present_cells = {
            (int(b["cell_x"]), int(b["cell_y"])) for b in pboxes
        }
        prod = rs.get("production") or []
        prod_items = [
            p.get("item") for p in prod if isinstance(p, dict)
        ]
        # Find the first rung that is currently uncovered (initial
        # build or post-attrition rebuild) and (build +) place there.
        for cell in rungs:
            if cell not in present_cells:
                cmds = []
                if "pbox" not in prod_items:
                    cmds.append(C.build("pbox"))
                cmds.append(C.place_building("pbox", cell[0], cell[1]))
                return cmds
        # All rungs currently covered — idle.
        return [C.observe()]

    return policy


def _wrong_topology_policy(cells):
    """Pile pboxes at a fixed list of cells (count-only, no rung
    rebuilding). Used for cluster-on-centre and scatter-near-base."""
    cells = list(cells)

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [
            p.get("item") for p in prod if isinstance(p, dict)
        ]
        if n >= len(cells):
            return [C.observe()]
        cmds = []
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", cells[n][0], cells[n][1]))
        return cmds

    return policy


def make_cluster_on_centre(level: str):
    """WRONG TOPOLOGY: K pboxes piled on the centre row (y≈20).
    Satisfies the count and the y=20 rung (if it exists) but misses
    every flank rung because the rung regions are radius 0.5 (cell-
    exact). The unguarded flank rows let the rush leak past."""
    return _wrong_topology_policy(CLUSTER_ON_CENTRE_BY_LEVEL[level])


def make_scatter_near_base(level: str):
    """WRONG TOPOLOGY: K pboxes hugging the fact west of x=20. Misses
    every front rung AND too far west to engage the rush before it
    reaches the fact (on harder tiers the flank rows reach the fact
    without ever encountering the LINE)."""
    return _wrong_topology_policy(SCATTER_NEAR_BASE_BY_LEVEL[level])


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-defensive-tower-line"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors.
    anchors = pack.meta.benchmark_anchor
    assert "ERQA" in anchors, anchors
    assert "MicroRTS defense" in anchors, anchors
    assert "military perimeter" in anchors, anchors
    # Rusher bot wired through (charges agent centroid → forces each
    # row's rush column WEST through the central x=60 LINE on every
    # seed).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_scales_per_tier_for_pbox_budget():
    """Cash is intentionally tight per tier — exactly K pboxes for
    easy (K=3) and medium (K=5), plus a 2-pbox rebuild margin for hard
    (K=6+2 rebuilds for the two-wave attrition) — so a model that
    spends on units OR extra rebuilds beyond the design cannot pass
    the count clause."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.starting_cash == CASH_BY_LEVEL[lvl], (
            lvl, c.starting_cash, CASH_BY_LEVEL[lvl]
        )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be strictly below the tick reachable at max_turns. No interrupts on
    this pack ⇒ each step is exactly 90 ticks (max tick = 93+90·(N-1))."""
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


def test_fact_alive_clause_uses_present_tense_predicate():
    """The fact-survival clause must use the PRESENT-TENSE predicate
    (`building_count_gte:{type:fact,n:1}`) rather than `has_building`,
    which is a one-shot "ever seen" set that stays true after the fact
    is destroyed (a documented CLAUDE.md footgun). Otherwise the rush
    razing the fact would not trigger a LOSS."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        fact_clauses = [
            clause for clause in fc.get("any_of", []) or []
            if isinstance(clause, dict)
            and isinstance(clause.get("not"), dict)
            and "building_count_gte" in (clause["not"] or {})
            and (clause["not"]["building_count_gte"] or {}).get("type") == "fact"
        ]
        assert fact_clauses, f"{lvl}: missing present-tense fact-alive fail clause"


@pytest.mark.parametrize("level", LEVELS)
def test_win_requires_one_pbox_per_front_rung(level):
    """The LINE-enforcement contract: each level's win clause requires
    exactly one pbox in EACH of the front rungs at x=60 spanning the
    full vertical width. A cluster on the centre row (y=20) misses
    every flank rung because each rung region has radius 0.5 (cell-
    exact). The rungs grow per tier (3 easy / 5 medium / 6 hard)."""
    expected = {y for (_, y) in RUNGS_BY_LEVEL[level]}
    c = compile_level(load_pack(PACK), level)
    wc = c.win_condition.model_dump(exclude_none=True)
    rungs_seen = set()
    for clause in wc.get("all_of", []) or []:
        br = clause.get("building_in_region")
        if (
            isinstance(br, dict)
            and br.get("type") == "pbox"
            and int(br.get("x", -1)) == 60
            and int(br.get("count", 0)) == 1
            and float(br.get("radius", 0)) <= 1.0
        ):
            rungs_seen.add(int(br["y"]))
    assert rungs_seen == expected, (
        f"{level}: front rungs y∈{sorted(expected)} required, got {sorted(rungs_seen)}"
    )


def test_win_requires_a_kill_quota():
    """The pbox LINE must actively KILL the rush, not merely stand: every
    level's win clause carries a `units_killed_gte` quota. With no
    pre-placed agent defenders the pbox LINE is the sole source of kills,
    so this clause makes the pbox weapon load-bearing."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        kill_clauses = [
            clause for clause in wc.get("all_of", []) or []
            if isinstance(clause, dict) and "units_killed_gte" in clause
        ]
        assert kill_clauses, f"{lvl}: missing units_killed_gte kill quota"
        assert int(kill_clauses[0]["units_killed_gte"]) >= 4, (lvl, kill_clauses)


def test_rush_arrives_as_a_scheduled_event():
    """The rush is injected via `scheduled_events: spawn_actors` AFTER the
    LINE has time to assemble — there is no t=0 enemy band racing the
    build. This is what makes the build/rush race fair. Hard tier has
    TWO scheduled waves (attrition mechanic)."""
    expected_wave_counts = {"easy": 1, "medium": 1, "hard": 2}
    for lvl in LEVELS:
        pack = load_pack(PACK)
        raw = pack.levels[lvl]
        ov = getattr(raw, "overrides", None) or {}
        if hasattr(ov, "model_dump"):
            ov = ov.model_dump(exclude_none=True)
        evts = ov.get("scheduled_events") or []
        assert evts, f"{lvl}: expected a scheduled rush wave"
        spawn_waves = [e for e in evts if e.get("type") == "spawn_actors"]
        assert spawn_waves, (lvl, evts)
        assert len(spawn_waves) == expected_wave_counts[lvl], (
            f"{lvl}: expected {expected_wave_counts[lvl]} spawn_actors waves, "
            f"got {len(spawn_waves)} ({evts})"
        )


def test_no_pre_placed_agent_combat_screen():
    """The pbox LINE must be the sole kill source — there is no
    pre-placed agent combat screen ringing the base. Only ONE
    non-combatant agent e1 is parked in a far corner (per spawn group)
    so units_summary is non-empty for the hard-tier env-reset check;
    it never fights."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        agent_units = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "e1"
        ]
        # At most one non-combatant marker per active spawn group
        # (hard has 2 spawn_point groups so up to 2 corner e1s
        # declared; only the active spawn group's e1 is materialised).
        assert len(agent_units) <= 2, (lvl, [a.position for a in agent_units])
        for a in agent_units:
            x, y = a.position
            # Parked in a far corner, well clear of the rush lanes
            # spread across y=4..36 at x=60.
            assert x <= 6 and (y <= 6 or y >= 34), (lvl, a.position)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so a
    memorised relative-to-base placement that lands in the same world
    cell on every seed cannot generalise."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check — the per-tier 112×48 wide-front arena
    # (cordon=2) has playable x ∈ [2, 109], y ∈ [2, 45].
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 109 and 2 <= y <= 45, (a.type, a.position)


# ── solvency: intended LINE wins every level + every hard seed ───────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_line_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_line(level), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended LINE topology must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / wrong-topology policy LOSES (not draws) ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall",              lambda lvl: stall),
        ("cluster_on_centre",  lambda lvl: make_cluster_on_centre(lvl)),
        ("scatter_near_base",  lambda lvl: make_scatter_near_base(lvl)),
    ],
)
def test_lazy_and_wrong_topology_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (rush razes fact AND clock runs out with no pbox), cluster-
    on-centre (count satisfied but every flank rung unmet), and scatter-
    near-base (every rung region unmet, rush reaches fact past unguarded
    front) must ALL LOSE on every level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory(level)
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_line("easy"), seed=3)
    b = run_level(c, make_line("easy"), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
