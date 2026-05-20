"""tp-survive-and-strike-at-window — temporal-sequencing pack.

The pack tests whether the agent can SURVIVE a pre-window (no engagement
before tick T1) and then STRIKE inside a narrow follow-up band [T1..T2].
The win predicate is the `then:` happened-before composite — clause 1
(survive: own_units_gte:4 at T1) must LATCH BEFORE clause 2 (strike:
units_killed_gte:K within T2) becomes evaluable.

Bar (binding):
- intended hold-then-strike-at-window policy WINS on every level + every
  hard seed (1..4);
- stall / strike-immediately / brute-rush LOSE on every level + every
  seed (real reachable timeout / survival-clause LOSS, no DRAW
  degeneracy);
- non-win is a real reachable timeout LOSS (within_ticks /
  after_ticks ≤ 93 + 90·(max_turns − 1));
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK = PACKS / "tp-survive-and-strike-at-window.yaml"


# ── 1) declarative / schema invariants (no engine needed) ──────────────


def test_pack_loads_and_three_levels_compile():
    p = load_pack(PACK)
    assert p.meta.id == "tp-survive-and-strike-at-window"
    assert p.meta.capability == "reasoning"
    # The three mandatory anchors from the authoring spec.
    anchors = " | ".join(p.meta.benchmark_anchor)
    for needed in (
        "SC2 timing push",
        "PlanBench temporally-extended",
        "cyber attack timing",
    ):
        assert needed in anchors, f"benchmark_anchor missing {needed!r}: {anchors}"
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        assert c.map_supported, f"{lv}: rush-hour-arena must be Rust-loadable"


def test_win_is_then_composite_with_two_clauses():
    """The win predicate must be the `then:` happened-before composite
    (clause 1 SURVIVAL gate → clause 2 STRIKE gate). An `all_of` would
    be satisfied by any state where both happen to be true now, which
    a kills-before-T1 plan could satisfy — defeating the timing axis."""
    p = load_pack(PACK)
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        assert "then" in wc, f"{lv}: win must use `then:` composite, got {list(wc)}"
        then_block = wc["then"]
        clauses = then_block.get("clauses") or []
        assert len(clauses) == 2, f"{lv}: then must have 2 clauses, got {len(clauses)}"
        # Clause 1: survival gate (after_ticks + own_units_gte).
        c1 = clauses[0]
        assert "all_of" in c1
        c1_keys = {k for cl in c1["all_of"] for k in cl}
        assert "after_ticks" in c1_keys and "own_units_gte" in c1_keys, (
            f"{lv}: survival clause must combine after_ticks + own_units_gte; got {c1}"
        )
        # Clause 2: strike gate (units_killed_gte + within_ticks).
        c2 = clauses[1]
        assert "all_of" in c2
        c2_keys = {k for cl in c2["all_of"] for k in cl}
        assert "units_killed_gte" in c2_keys and "within_ticks" in c2_keys, (
            f"{lv}: strike clause must combine units_killed_gte + within_ticks; got {c2}"
        )


def test_window_tightens_across_tiers():
    """Difficulty axis = window width (T2 - T1) shrinking + kill bar K
    rising. One controlled variable family per tier."""
    p = load_pack(PACK)
    widths = {}
    ks = {}
    t1s = {}
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        clauses = wc["then"]["clauses"]
        t1 = next(cl["after_ticks"] for cl in clauses[0]["all_of"] if "after_ticks" in cl)
        t2 = next(cl["within_ticks"] for cl in clauses[1]["all_of"] if "within_ticks" in cl)
        k = next(cl["units_killed_gte"] for cl in clauses[1]["all_of"] if "units_killed_gte" in cl)
        widths[lv] = t2 - t1
        ks[lv] = k
        t1s[lv] = t1
    # Window must monotonically narrow.
    assert widths["easy"] > widths["medium"] >= widths["hard"], (
        f"window widths must narrow easy→hard: {widths}"
    )
    # Pre-window hold must monotonically lengthen (later T1).
    assert t1s["easy"] <= t1s["medium"] <= t1s["hard"], (
        f"T1 must non-decrease easy→hard: {t1s}"
    )
    # Kill bar must non-decrease.
    assert ks["easy"] <= ks["medium"] <= ks["hard"], (
        f"K must non-decrease easy→hard: {ks}"
    )


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
def test_within_and_after_ticks_reachable_no_draw_degeneracy(lv):
    """`within_ticks` (win) and the fail `after_ticks` must both be
    reachable within `max_turns` (tick ≤ 93 + 90·(max_turns − 1)) or
    the episode would time out as a DRAW instead of a real LOSS."""
    p = load_pack(PACK)
    c = compile_level(p, lv)
    ceiling = 93 + 90 * (c.max_turns - 1)
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    clauses = wc["then"]["clauses"]
    wt = next(cl["within_ticks"] for cl in clauses[1]["all_of"] if "within_ticks" in cl)
    assert wt < ceiling, f"{lv}: within_ticks {wt} ≥ ceiling {ceiling} ⇒ inert"
    fc = dict(c.fail_condition.__pydantic_extra__ or {})
    aft = next(cl["after_ticks"] for cl in fc["any_of"] if "after_ticks" in cl)
    assert aft <= ceiling, (
        f"{lv}: fail after_ticks {aft} > ceiling {ceiling} ⇒ unreachable ⇒ DRAW"
    )
    # The timeout fail bites exactly the tick AFTER the strike deadline,
    # so an agent that misses the window loses immediately (no
    # one-turn ambiguity).
    assert aft == wt + 1, (
        f"{lv}: fail after_ticks {aft} should be {wt + 1} (the tick after win)"
    )


def test_hard_has_multiple_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct seed-driven spawn groups so a
    memorised opening cannot generalise."""
    p = load_pack(PACK)
    c = compile_level(p, "hard")
    sp = {
        a.spawn_point if a.spawn_point is not None else 0
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sorted(sp)}"


def test_enemy_is_turtle_bot():
    """`turtle` bot keeps the enemy from chasing the holding agent during
    the pre-window, so the pre-window hold is genuinely safe (the
    discriminator is the agent's timing, not whether it gets ambushed
    on staging)."""
    p = load_pack(PACK)
    assert p.base.get("enemy", {}).get("bot_type") == "turtle"


def test_fail_has_base_loss_and_force_loss_clauses():
    """Fail tree must trigger on (a) timeout, (b) base loss
    (`has_building: fact` violated), (c) total force wipe (gated with
    after_ticks per CLAUDE.md not-own_units-gte:1 turn-1 footgun)."""
    p = load_pack(PACK)
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        anyof = fc["any_of"]
        keys_flat: set[str] = set()
        for cl in anyof:
            keys_flat |= set(cl)
            if "not" in cl:
                keys_flat |= set(cl["not"])
            if "all_of" in cl:
                for sub in cl["all_of"]:
                    keys_flat |= set(sub)
                    if "not" in sub:
                        keys_flat |= set(sub["not"])
        assert "after_ticks" in keys_flat, f"{lv}: fail must include timeout clause"
        assert "has_building" in keys_flat, f"{lv}: fail must include fact-loss clause"
        assert "own_units_gte" in keys_flat, f"{lv}: fail must include force-wipe clause"


# ── 2) engine-required scripted-policy sweep ───────────────────────────


def _stall(rs, C):
    """Never moves — must LOSE on the timeout (kill bar unmet)."""
    return [C.observe()]


def _strike_immediately(rs, C):
    """All tanks attack-move to the central enemy cluster from tick 0.
    Crosses the tsla + 3tnk envelope before T1 and loses at least one
    tank → survival clause `own_units_gte:4` is broken before it can
    latch → the `then:` chain never advances → real LOSS at T2+1."""
    us = rs.get("units_summary") or []
    ids = [str(u["id"]) for u in us]
    if not ids:
        return [C.observe()]
    return [C.attack_move(ids, 58, 20)]


def _intended_hold_then_strike_easy(rs, C):
    """Hold at staging until tick ~1250 (just past T1=1200), then
    attack-move the full column at the central cluster. The kill bar
    K=3 should fall well within T2=4200."""
    return _intended(rs, C, hold_until_tick=1250)


def _intended_hold_then_strike_medium(rs, C):
    """T1=1500, K=4, T2=3600."""
    return _intended(rs, C, hold_until_tick=1550)


def _intended_hold_then_strike_hard(rs, C):
    """T1=1800, K=4, T2=3300."""
    return _intended(rs, C, hold_until_tick=1850)


def _intended(rs, C, *, hold_until_tick: int):
    """Hold (observe) until `hold_until_tick`, then attack-move the
    full tank column at the central enemy cluster (58, 20)."""
    tick = int(rs.get("game_tick", 0) or 0)
    us = rs.get("units_summary") or []
    # Filter to combat units only (the 4× 2tnk — the column).
    ids = [str(u["id"]) for u in us if str(u.get("type", "")).lower() == "2tnk"]
    if tick < hold_until_tick or not ids:
        return [C.observe()]
    return [C.attack_move(ids, 58, 20)]


_INTENDED = {
    "easy": _intended_hold_then_strike_easy,
    "medium": _intended_hold_then_strike_medium,
    "hard": _intended_hold_then_strike_hard,
}

# (name, fn) — every "wrong" policy must LOSE on every (level, seed).
_NEGATIVE = [
    ("stall", _stall),
    ("strike-immediately", _strike_immediately),
]


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_hold_then_strike_wins(lv, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, _INTENDED[lv], seed=seed)
    assert r.outcome == "win", (
        f"{lv}:seed{seed} intended hold-then-strike must WIN "
        f"(got {r.outcome} at tick {r.signals.game_tick}, "
        f"killed={r.signals.units_killed}, lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("policy_name,policy_fn", _NEGATIVE)
@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_negative_policy_loses(lv, seed, policy_name, policy_fn):
    """stall / strike-immediately: every level, every seed, REAL
    reachable LOSS (no DRAW degeneracy)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, policy_fn, seed=seed)
    assert r.outcome == "loss", (
        f"{lv}:seed{seed} {policy_name} must LOSE on the bar "
        f"(got {r.outcome} at tick {r.signals.game_tick}, "
        f"killed={r.signals.units_killed}, lost={r.signals.units_lost})"
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
    assert len(starts) >= 2, (
        f"hard seeds produced identical starts {starts}; "
        "spawn round-robin off"
    )
