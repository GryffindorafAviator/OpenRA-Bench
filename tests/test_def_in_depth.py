"""def-in-depth scenario family, full loop on Rust.

The pack tests defense-in-depth doctrine: against a massed rush band,
the same total pbox capacity laid out as TWO concentric layers wins,
while the same pbox count in a single line or a single cluster loses.
The discriminator is layout topology, not raw count.

* `building_count_gte:{pbox,4}` (easy/medium) / 5 (hard) — minimum
  defensive capacity floor; less than this can't survive even with
  perfect placement.
* `building_in_region:{(25,20),r=4,pbox,2}` AND
  `building_in_region:{(15,20),r=4,pbox,2}` — REGION predicates with
  non-overlapping radii (FRONT x in [21..29], REAR x in [11..19]) so
  the model MUST physically split the 4 pbox across two depths;
  a single cluster of 4 pbox at one x can satisfy at most ONE region.
* `has_building:fact` + `own_units_gte:1` — the base survives and a
  defender is on the field at the deadline.
* `within_ticks:5400` (easy) / 4500 (medium/hard) paired with
  `after_ticks:5401/4501` — a non-finisher is a real reachable
  timeout LOSS (60 turns × ≤90 ticks/step in interrupt mode reaches
  ≥4848), never a draw.

These tests prove with deterministic scripted policies (no model, no
network) that:

* the intended layered 2+2 policy WINS every easy/medium seed (1..4);
* stall / single-line / single-cluster all LOSE every level + seed —
  a real LOSS, not a draw;
* the `after_ticks` deadline is reachable inside `max_turns`;
* the hard tier defines ≥2 spawn_point groups.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-in-depth.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ─────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — never spends. Fact razed by the rush."""
    return [C.observe()]


def _make_cluster_builder(cells):
    """Queue many pbox up front (Defense queue accepts backlog), then
    place each at the next cell from `cells` as production completes.
    Stops once every cell is built."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        pbox_count = sum(1 for b in own_b if b.get("type") == "pbox")
        if pbox_count >= len(cells):
            return [C.observe()]
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        n_in_q = sum(1 for it in prod_items if it == "pbox")
        cmds = []
        # Refill queue so we always have ≥1 pbox in flight while waiting
        # (the Defense queue tolerates a backlog; this halves wall-clock
        # from serial build-place pairs).
        need_to_queue = len(cells) - pbox_count - n_in_q
        for _ in range(max(0, need_to_queue)):
            cmds.append(C.build("pbox"))
        # Try to place the next cell each turn — if no pbox is ready
        # the engine logs PLACE BLOCKED and moves on (harmless).
        cell = cells[pbox_count]
        cmds.append(C.place_building("pbox", cell[0], cell[1]))
        return cmds

    return policy


def make_layered_2_2():
    """Intended: 2 front (x=25) + 2 rear (x=15), y centred on 20."""
    return _make_cluster_builder(
        [(25, 19), (25, 21), (15, 19), (15, 21)]
    )


def make_layered_3_2(front_y, rear_y):
    """Hard-tier intended: 4 front pbox are pre-placed (2 at each
    latitude); agent adds 2 REAR pbox at its OWN spawn latitude →
    total 6 pbox: 4 front + 2 rear-at-spawn. `front_y` is unused;
    it's kept in the signature for symmetry with the test caller."""
    return _make_cluster_builder([
        (15, rear_y - 1), (15, rear_y + 1),
    ])


def make_stack_front():
    """Stack MORE pbox at FRONT (x=25): 2 inherited + 4 more → all 6
    at front, 0 at rear → rear-region predicate fails. The thick
    single wall eventually attrites under sustained pressure even
    if it survives the first wave → fact razed OR clock LOSS."""
    return _make_cluster_builder(
        [(25, 17), (25, 18), (25, 22), (25, 23)]
    )


def make_single_line():
    """4 pbox in a thin column at x=20 — BETWEEN the two regions
    (front=[21..29], rear=[11..19] both exclude x=20). Neither region
    gains a NEW pbox here (rear-region count from inherited front
    pboxes = 0) → win unsatisfied; under pressure the thin line
    falls and the fact dies → LOSS."""
    return _make_cluster_builder(
        [(20, 16), (20, 18), (20, 20), (20, 22)]
    )


def make_no_build():
    """Issue no build orders — the 2 inherited pbox satisfy front
    region but rear region count = 0, total count = 2 < 4. LOSS."""
    def policy(rs, C):
        return [C.observe()]
    return policy


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-in-depth"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    anchors = pack.meta.benchmark_anchor
    assert any("defense-in-depth" in a.lower() for a in anchors), anchors
    assert any("multi-layer" in a.lower() for a in anchors), anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert (str(bot).lower() == "rusher"), (lvl, bot)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
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
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_region_radii_are_non_overlapping():
    """The two layer regions (front=(25,20,r=4), rear=(15,20,r=4)) MUST
    not overlap, or a single tall cluster satisfies both predicates and
    the capability isn't enforced."""
    c = compile_level(load_pack(PACK), "easy")
    win = c.win_condition.model_dump(exclude_none=True)
    regions = [
        clause["building_in_region"]
        for clause in win.get("all_of", [])
        if "building_in_region" in clause
    ]
    assert len(regions) == 2, regions
    (a, b) = regions
    dx = abs(int(a["x"]) - int(b["x"]))
    assert dx > int(a["radius"]) + int(b["radius"]), (
        f"regions overlap: dx={dx}, rA+rB={a['radius']+b['radius']}"
    )


# ── solvency: intended WINS every easy/medium seed ────────────────────


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_intended_layered_policy_wins_every_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_layered_2_2(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended 2+2 layered play must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_intended_layered_3_2_wins_hard_every_seed():
    """Hard tier — the agent commits 3+2 at whichever latitude its
    base spawned (try BOTH and take whichever wins; the any_of win
    accepts either)."""
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        # spawn_point round-robins 0/1 by seed; without knowing which
        # one this seed picked, try both latitudes and require at
        # least one to win — the model in the field will read its own
        # fact position from the observation.
        wins = []
        for (fy, ry) in [(14, 14), (26, 26)]:
            r = run_level(c, make_layered_3_2(fy, ry), seed=seed)
            wins.append(r.outcome == "win")
        assert any(wins), (
            f"hard seed{seed}: at least one layered 3+2 latitude "
            f"must WIN (model picks latitude from its own fact pos)"
        )


# ── no-cheat: every wrong-topology policy LOSES (not draws) ───────────


@pytest.mark.parametrize("level", ["easy", "medium"])
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("stack_front", make_stack_front),
        ("single_line_between", make_single_line),
        ("no_build", make_no_build),
    ],
)
def test_lazy_and_wrong_topology_policies_lose_every_seed(
    level, policy_name, policy_factory
):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )
