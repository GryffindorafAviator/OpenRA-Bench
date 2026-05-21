"""mid-concede-vs-hold pack: validate the concede-vs-hold idiom.

The bar (per CLAUDE.md): every lazy / brute / split / oscillating /
stall policy must LOSE on every level + every hard seed; the intended
consolidate-on-the-light-side policy must WIN; the timeout is a real
reachable LOSS. This file checks each of those, the schema-level
properties (spawn_point contract for hard, fail_condition shape,
benchmark anchors), and that the win/fail predicate tree is in the
right band (after_ticks ≤ within_ticks ≤ reachable-tick).

Recalibration note: the engine combat rebalance hugely strengthened
stationary defenders, breaking the original bar — the light/heavy
pushes (4 / 8-12 rifles) no longer threatened the buffed garrisons,
and an `enemy_units_killed` auto-`done` ended the episode (DRAW)
the instant the agent cleared a push, before the survival floor.
The pack was re-tuned: pushes scaled up (WEST 16 / EAST 36-42 /
hard EAST 34), `enemy_units_killed` termination dropped (the win is
a survival-band check), a `not proc:1` fail clause added (a
wrong-side consolidate loses every refinery — without it that play
kept a lone fact and silently DREW), a persistent unarmed enemy
`fact` marker added (anti auto-DRAW), the hard survival floor moved
to tick 2400 and its attrition cap to 18. The capability stays
load-bearing — stall / split / oscillate / wrong-side consolidate
all LOSE; only consolidate-on-the-light-side WINS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_FILE = PACKS / "mid-concede-vs-hold.yaml"


# ── Schema / static properties ──────────────────────────────────────────


def test_pack_loads_and_compiles_all_levels():
    p = load_pack(PACK_FILE)
    assert p.meta.id == "mid-concede-vs-hold"
    assert p.meta.capability == "reasoning"
    # Benchmark anchors required by spec.
    anchors = [a.lower() for a in p.meta.benchmark_anchor]
    assert any("planbench" in a for a in anchors)
    assert any("cicero" in a or "diplomacy" in a for a in anchors)
    assert any("triage" in a or "outage" in a for a in anchors)
    assert any("datacentre" in a or "datacenter" in a or "failover" in a for a in anchors)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(p, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def test_within_ticks_reachable_in_max_turns():
    """No draw degeneracy: every level's fail `after_ticks` must be
    reachable within max_turns (engine advances ~90 ticks/turn,
    tick ≈ 93 + 90·(max_turns − 1))."""
    p = load_pack(PACK_FILE)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(p, lvl)
        # Pull the after_ticks fail leaf (loss-on-timeout). Each level's
        # fail_condition.any_of[0] is `{after_ticks: K}`.
        fail_dict = dict(c.fail_condition.__pydantic_extra__ or {})
        any_of = fail_dict.get("any_of") or []
        after_leaf = next(
            (
                dict(getattr(n, "__pydantic_extra__", {}) or {}) if not isinstance(n, dict) else n
                for n in any_of
                if (isinstance(n, dict) and "after_ticks" in n)
                or (
                    not isinstance(n, dict)
                    and "after_ticks" in (getattr(n, "__pydantic_extra__", {}) or {})
                )
            ),
            None,
        )
        assert after_leaf is not None, f"{lvl}: fail must include a timeout after_ticks leaf"
        K = int(after_leaf["after_ticks"])
        reachable = 93 + 90 * (c.max_turns - 1)
        assert K <= reachable, (
            f"{lvl}: after_ticks={K} not reachable within max_turns={c.max_turns} "
            f"(max reachable tick ≈ {reachable}); would degenerate to a DRAW"
        )


def test_hard_has_two_spawn_point_groups_for_seed_variation():
    """Hard-tier contract (see tests/test_hard_tier.py::UPGRADED): the
    hard level must define ≥2 distinct agent spawn_point groups so the
    engine round-robins the start per seed (rotating which side the
    flex squad is pre-positioned closer to)."""
    c = compile_level(load_pack(PACK_FILE), "hard")
    sps = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    sps.discard(None)
    assert sps == {0, 1}, f"hard agent spawn_points expected {{0,1}}, got {sps}"


def test_hard_objective_coords_relative():
    """Hard tier hides exact coordinates in the briefing — the model
    must localise the saveable base on the minimap itself."""
    c = compile_level(load_pack(PACK_FILE), "hard")
    assert c.objective_coords == "relative"


def test_actors_in_map_bounds():
    """Every actor must be inside the generated arena's playable
    bounds (160x60 cordon 4 ⇒ x∈[4,155], y∈[4,55]) — actors placed
    off-map panic the engine."""
    p = load_pack(PACK_FILE)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(p, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 4 <= x <= 155, f"{lvl}: {a.type} at ({x},{y}) x out of bounds"
            assert 4 <= y <= 55, f"{lvl}: {a.type} at ({x},{y}) y out of bounds"


# ── Scripted policy sweep (deterministic, no model) ─────────────────────


pytestmark_engine = pytest.mark.skipif(  # type: ignore[attr-defined]
    pytest.importorskip("openra_train", reason="rust engine not built") is None,
    reason="rust engine not built",
)


def _stall(rs, C):
    return [C.observe()]


def _cons_west(rs, C):
    """Intended policy: commit every mobile unit to the WEST base
    (the light-push side). East garrison moves west too."""
    u = rs.get("units_summary", []) or []
    cmds = [
        C.move_units([str(x["id"])], target_x=26, target_y=20)
        for x in u
        if x["cell_x"] > 35
    ]
    return cmds or [C.observe()]


def _cons_east(rs, C):
    """Wrong-side commit: everything to east (heavy-push side)."""
    u = rs.get("units_summary", []) or []
    cmds = [
        C.move_units([str(x["id"])], target_x=76, target_y=20)
        for x in u
        if x["cell_x"] < 65
    ]
    return cmds or [C.observe()]


def _split_defend(rs, C):
    """Split-defend: half the flex squad to each base, garrisons stay."""
    u = rs.get("units_summary", []) or []
    flex = sorted(
        [x for x in u if 38 <= x["cell_x"] <= 65], key=lambda x: x["id"]
    )
    cmds = []
    for i, x in enumerate(flex):
        tgt = (26, 20) if i % 2 == 0 else (76, 20)
        cmds.append(C.move_units([str(x["id"])], target_x=tgt[0], target_y=tgt[1]))
    return cmds or [C.observe()]


class _PanicTC:
    def __init__(self):
        self.n = 0


def _panic_reinforce_factory():
    """Reinforce-both-back-and-forth: flip the flex target every turn
    (no commitment, no rest). Pure wasted travel."""
    tc = _PanicTC()

    def f(rs, C):
        tc.n += 1
        u = rs.get("units_summary", []) or []
        flex = [
            x
            for x in u
            if 38 <= x["cell_x"] <= 65
            or abs(x["cell_x"] - 26) < 5
            or abs(x["cell_x"] - 76) < 5
        ]
        tgt = (26, 20) if tc.n % 2 == 0 else (76, 20)
        return [
            C.move_units([str(x["id"])], target_x=tgt[0], target_y=tgt[1])
            for x in flex
        ] or [C.observe()]

    return f


def _run(level: str, policy, seed: int = 1):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_FILE), level)
    return run_level(c, policy, seed=seed)


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_intended_consolidate_west_wins(lvl):
    """The intended consolidate-on-the-light-side policy must WIN on
    every level. On hard this is run across every seed (1..4) — the
    weak-side commit must work regardless of which spawn group the
    flex squad starts in."""
    seeds = (1, 2, 3, 4) if lvl == "hard" else (1,)
    for seed in seeds:
        res = _run(lvl, _cons_west, seed=seed)
        assert res.outcome == "win", (
            f"{lvl} seed={seed}: intended consolidate-west must WIN, got "
            f"{res.outcome} at tick {res.signals.game_tick}"
        )


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_stall_loses(lvl):
    """Doing nothing must LOSE on every level (both bases razed under
    the clock — no draw degeneracy)."""
    seeds = (1, 2, 3, 4) if lvl == "hard" else (1,)
    for seed in seeds:
        res = _run(lvl, _stall, seed=seed)
        assert res.outcome == "loss", (
            f"{lvl} seed={seed}: stall must LOSE, got {res.outcome}"
        )


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_split_defend_loses(lvl):
    """Split-defend (half flex to each base) must LOSE — neither side
    gets enough reinforcements; both bases fall."""
    seeds = (1, 2, 3, 4) if lvl == "hard" else (1,)
    for seed in seeds:
        res = _run(lvl, _split_defend, seed=seed)
        assert res.outcome == "loss", (
            f"{lvl} seed={seed}: split-defend must LOSE, got {res.outcome}"
        )


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_reinforce_back_and_forth_loses(lvl):
    """Reinforce-both-back-and-forth (panic flipping every turn) must
    LOSE — pure wasted travel, flex never settles long enough to
    actually defend."""
    seeds = (1, 2, 3, 4) if lvl == "hard" else (1,)
    for seed in seeds:
        res = _run(lvl, _panic_reinforce_factory(), seed=seed)
        assert res.outcome == "loss", (
            f"{lvl} seed={seed}: reinforce-back-and-forth must LOSE, got "
            f"{res.outcome}"
        )


def test_commit_to_heavy_side_loses_medium():
    """Wrong-side commit (consolidate EAST, the heavy push) must
    LOSE: the heavy push is too large for any defence; both bases
    fall."""
    res = _run("medium", _cons_east, seed=1)
    assert res.outcome == "loss", (
        f"commit-to-heavy must LOSE on medium, got {res.outcome}"
    )


# ── Hard-tier spawn rotation (engine round-robins by seed) ──────────────


def test_hard_seeds_produce_different_flex_starts():
    """Hard tier seeds 1..4 must actually place the flex squad at
    different starting positions (the whole point of spawn_point
    rotation — anti-memorisation)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    c = compile_level(load_pack(PACK_FILE), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    try:
        starts = set()
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            # Pick the flex-squad x-coords (around 42 or 60 by design).
            flex_x = tuple(sorted({x["cell_x"] for x in u if 38 <= x["cell_x"] <= 65}))
            if flex_x:
                starts.add(flex_x)
        assert len(starts) >= 2, (
            f"hard: seed-driven spawn variation did not take effect; "
            f"distinct flex starts = {starts}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
