"""coordination-staggered-window — parallel-scheduling pack.

The pack tests whether the controller can DISPATCH IN PARALLEL so that
EVERY dock is staffed at the same instant before a shared deadline. The
"staggered window" semantic is captured by per-region landmark distances
that force the long-haul team to launch first; a stall, a brute single-
region rush, and a single-column tour all FAIL by construction.

Bar (binding):
- intended split-and-dispatch policy WINS on every level + every hard
  seed (1..4);
- stall / brute / single-column-tour LOSE on every level + every seed;
- the non-win is a real reachable timeout LOSS (no DRAW degeneracy:
  `within_ticks` ≤ 93 + 90·(max_turns − 1));
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK = PACKS / "coordination-staggered-window.yaml"


# ── 1) declarative / schema invariants (no engine needed) ──────────────

def test_pack_loads_and_three_levels_compile():
    p = load_pack(PACK)
    assert p.meta.id == "coordination-staggered-window"
    assert p.meta.capability == "action"
    # Authoring spec: at least four named anchors (Watch-And-Help,
    # SMAC, MARLÖ, multi-robot dispatch) — every one is mandatory.
    anchors = " | ".join(p.meta.benchmark_anchor)
    for needed in ("Watch-And-Help", "SMAC", "MARLÖ", "multi-robot"):
        assert needed in anchors, f"benchmark_anchor missing {needed!r}: {anchors}"
    assert "warehouse" in p.meta.real_world_meaning.lower()
    assert "multi-robot" in p.meta.robotics_analogue.lower()
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        assert c.map_supported, f"{lv}: rush-hour-arena must be Rust-loadable"


def test_win_enforces_split_via_units_in_region_gte():
    """The split must be enforced — every region clause is `units_in_
    region_gte` with n ≥ 3, never `reach_region` (the classic "one
    touring unit wins" inversion)."""
    p = load_pack(PACK)
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        clauses = wc["all_of"]
        regs = [cl for cl in clauses if "units_in_region_gte" in cl]
        assert len(regs) >= 2, f"{lv}: need ≥2 region clauses, got {regs}"
        assert all(cl["units_in_region_gte"].get("n", 1) >= 3 for cl in regs), \
            f"{lv}: every region must require n≥3 (split), got {regs}"
        assert not any("reach_region" in cl for cl in clauses), \
            f"{lv}: reach_region (n≥1) would let one touring unit win"


def test_hard_has_three_regions_and_relative_coords():
    """Hard adds the 3rd region + coordinate-blind grounding."""
    p = load_pack(PACK)
    c = compile_level(p, "hard")
    assert c.objective_coords == "relative"
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    regs = [cl for cl in wc["all_of"] if "units_in_region_gte" in cl]
    assert len(regs) == 3, f"hard must have 3 regions, got {len(regs)}"
    # Relative-mode regions need an authored compass label, else the
    # primer would silently leak coordinates.
    for cl in regs:
        assert cl["units_in_region_gte"].get("label"), \
            f"hard region missing 'label' for relative-coords brief: {cl}"


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
def test_within_ticks_is_reachable_no_draw_degeneracy(lv):
    """`within_ticks` and the fail `after_ticks` must both be reachable
    within `max_turns` (tick ≤ 93 + 90·(max_turns − 1)) or the episode
    would time out as a DRAW instead of a real LOSS."""
    p = load_pack(PACK)
    c = compile_level(p, lv)
    ceiling = 93 + 90 * (c.max_turns - 1)
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    wt = next(cl["within_ticks"] for cl in wc["all_of"] if "within_ticks" in cl)
    assert wt < ceiling, f"{lv}: within_ticks {wt} ≥ ceiling {ceiling} ⇒ inert"
    fc = dict(c.fail_condition.__pydantic_extra__ or {})
    aft = next(cl["after_ticks"] for cl in fc["any_of"] if "after_ticks" in cl)
    assert aft <= ceiling, \
        f"{lv}: fail after_ticks {aft} > ceiling {ceiling} ⇒ unreachable ⇒ DRAW"
    # A staller / wrong-policy must be able to hit the deadline as a
    # real LOSS (the bar's "non-win is a reachable timeout LOSS").
    assert aft == wt + 1, \
        f"{lv}: fail after_ticks {aft} should be {wt + 1} (the tick after win)"


def test_hard_has_multiple_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct seed-driven spawn groups so a
    memorised opening cannot generalise."""
    p = load_pack(PACK)
    c = compile_level(p, "hard")
    sp = {a.spawn_point if a.spawn_point is not None else 0
          for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sorted(sp)}"


# ── 2) engine-required scripted-policy sweep ───────────────────────────

def _stall(rs, C):
    return [C.observe()]


def _brute(rs, C):
    """All units one-region rush (NE dock A). Only ever satisfies one
    of the n region clauses — must LOSE."""
    ids = [str(u["id"]) for u in (rs.get("units_summary") or [])]
    return [C.move_units(ids, 115, 6)] if ids else [C.observe()]


def _single_column_tour(rs, C):
    """Single column visits dock A then continues to dock B. At the
    instant the column is at B, A is empty, so the joint AND-clause
    never co-fires before the deadline — must LOSE."""
    us = rs.get("units_summary") or []
    ids = [str(u["id"]) for u in us]
    if not ids:
        return [C.observe()]
    in_a = sum(1 for u in us
               if (u["cell_x"] - 115) ** 2 + (u["cell_y"] - 6) ** 2 <= 64)
    target = (20, 36) if in_a >= 1 else (115, 6)
    return [C.move_units(ids, *target)]


def _intended_2(rs, C):
    """Split-and-dispatch — half east to NE dock, half SW to dock B.
    The intended policy for easy + medium."""
    us = rs.get("units_summary") or []
    ids = [str(u["id"]) for u in us]
    if len(ids) < 2:
        return [C.observe()]
    h = len(ids) // 2
    return [C.move_units(ids[:h], 115, 6),
            C.move_units(ids[h:], 20, 36)]


def _intended_3(rs, C):
    """Three-way split — NE / SW / SE. The intended policy for hard."""
    us = rs.get("units_summary") or []
    ids = [str(u["id"]) for u in us]
    if len(ids) < 3:
        return [C.observe()]
    t = len(ids) // 3
    return [C.move_units(ids[:t], 115, 6),
            C.move_units(ids[t:2 * t], 20, 36),
            C.move_units(ids[2 * t:], 115, 34)]


# (level, intended_policy_fn) — the appropriate intended policy per tier.
_INTENDED = {"easy": _intended_2, "medium": _intended_2, "hard": _intended_3}

# (name, fn) — every "wrong" policy must LOSE on every (level, seed).
_NEGATIVE = [
    ("stall", _stall),
    ("brute", _brute),
    ("tour", _single_column_tour),
]


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_policy_wins(lv, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, _INTENDED[lv], seed=seed)
    assert r.outcome == "win", \
        f"{lv}:seed{seed} intended split-dispatch must WIN (got {r.outcome})"


@pytest.mark.parametrize("policy_name,policy_fn", _NEGATIVE)
@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_negative_policy_loses(lv, seed, policy_name, policy_fn):
    """stall / brute / single-column-tour: every level, every seed,
    REAL reachable timeout LOSS (no DRAW degeneracy)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, policy_fn, seed=seed)
    assert r.outcome == "loss", (
        f"{lv}:seed{seed} {policy_name} must LOSE on the deadline "
        f"(got {r.outcome} at tick {r.signals.game_tick})"
    )


def test_hard_seeds_produce_distinct_starts():
    """The two `spawn_point` groups must actually round-robin under
    seeds 1..4 (the contract enforced for `UPGRADED` hard tiers)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import _scenario_to_tmp_yaml, RustEnvPool
    from openra_bench.rust_adapter import RustObsAdapter
    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts: set = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary") or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, \
        f"hard seeds produced identical starts {starts}; spawn round-robin off"
