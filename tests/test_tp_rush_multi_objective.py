"""tp-rush-multi-objective — ACTION parallel-scheduling pack.

The pack tests PARALLEL SCHEDULING (speedrun parallel / parallel
scheduling / SC2 multi-prong anchor): a strike force of 4× 2tnk + 2×
jeep is pre-staged at the far west as TWO aligned sub-groups, and the
agent must raze TWO enemy `fact`s — one at the NE corner (115,8), one
at the SE corner (115,32) — inside one tight shared clock. The
intended play dispatches BOTH prongs in the same decision turn so the
two ~1100-tick west→east rushes run concurrently; a serial play
(raze one objective, then dispatch the other prong only after the
first fact falls) stacks the rushes back-to-back (~2× make-span) and
blows the clock.

Bar (binding):
- intended PARALLEL dispatch WINS on every level + every hard seed
  (1..4);
- stall LOSES on every level + every seed (real reachable timeout
  LOSS, no DRAW degeneracy);
- a SERIAL play (raze one objective, then dispatch the other prong)
  LOSES on every level + every seed — the stacked rushes bust the
  clock;
- non-win is a real reachable timeout LOSS
  (within_ticks / after_ticks ≤ 93 + 90·(max_turns − 1));
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK = PACKS / "tp-rush-multi-objective.yaml"

NE = (115, 8)
SE = (115, 32)


# ── 1) declarative / schema invariants (no engine needed) ──────────────


def test_pack_loads_and_three_levels_compile():
    p = load_pack(PACK)
    assert p.meta.id == "tp-rush-multi-objective"
    assert p.meta.capability == "action"
    anchors = " | ".join(p.meta.benchmark_anchor)
    for needed in ("speedrun parallel", "parallel scheduling", "SC2 multi-prong"):
        assert needed in anchors, f"benchmark_anchor missing {needed!r}: {anchors}"
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        assert c.map_supported, f"{lv}: rush-hour-arena must be Rust-loadable"


def test_win_requires_two_region_scoped_fact_destructions():
    """Win = TWO `enemy_key_buildings_destroyed_in_region` clauses (NE
    + SE), each types=[fact], plus `within_ticks`. The two region
    clauses are what enforce 'BOTH far-apart objectives fell' — the
    type-only predicate would be satisfied by razing a single fact."""
    p = load_pack(PACK)
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        assert "all_of" in wc, f"{lv}: win must be `all_of`"
        clauses = wc["all_of"]
        regions = [
            cl["enemy_key_buildings_destroyed_in_region"]
            for cl in clauses
            if "enemy_key_buildings_destroyed_in_region" in cl
        ]
        assert len(regions) == 2, (
            f"{lv}: win must have exactly TWO region-scoped fact "
            f"destruction clauses; got {len(regions)}"
        )
        coords = {(int(r["x"]), int(r["y"])) for r in regions}
        assert coords == {NE, SE}, f"{lv}: region coords must be NE+SE; got {coords}"
        for r in regions:
            assert "fact" in {str(t).lower() for t in r["types"]}, (
                f"{lv}: region types must include 'fact'"
            )
        assert any("within_ticks" in cl for cl in clauses), (
            f"{lv}: win must include `within_ticks` (speedrun teeth)"
        )


def test_clock_tightens_easy_to_medium_hard():
    """easy = loose clock; medium/hard = tighter clock (same value —
    hard's +variable is spawn variation, not a further clock cut)."""
    p = load_pack(PACK)
    wins: dict[str, int] = {}
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        wins[lv] = next(
            cl["within_ticks"] for cl in wc["all_of"] if "within_ticks" in cl
        )
    assert wins["easy"] > wins["medium"], f"medium must tighten the clock: {wins}"
    assert wins["medium"] == wins["hard"], (
        f"hard's +variable is spawn variation; clock matches medium: {wins}"
    )


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
def test_within_and_after_ticks_reachable_no_draw_degeneracy(lv):
    """`within_ticks` (win) and the fail `after_ticks` must both be
    reachable within `max_turns` (tick ≤ 93 + 90·(max_turns − 1)) or
    a staller times out as a DRAW instead of a real LOSS."""
    p = load_pack(PACK)
    c = compile_level(p, lv)
    ceiling = 93 + 90 * (c.max_turns - 1)
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    wt = next(cl["within_ticks"] for cl in wc["all_of"] if "within_ticks" in cl)
    assert wt < ceiling, f"{lv}: within_ticks {wt} ≥ ceiling {ceiling} ⇒ inert"
    fc = dict(c.fail_condition.__pydantic_extra__ or {})
    aft = next(cl["after_ticks"] for cl in fc["any_of"] if "after_ticks" in cl)
    assert aft <= ceiling, (
        f"{lv}: fail after_ticks {aft} > ceiling {ceiling} ⇒ unreachable ⇒ DRAW"
    )
    assert aft == wt + 1, f"{lv}: fail after_ticks should be within_ticks+1"


def test_fail_has_timeout_and_base_loss_clauses():
    p = load_pack(PACK)
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        keys_flat: set[str] = set()
        for cl in fc["any_of"]:
            keys_flat |= set(cl)
            if "not" in cl:
                keys_flat |= set(cl["not"])
        assert "after_ticks" in keys_flat, f"{lv}: fail must include timeout clause"
        assert "building_count_gte" in keys_flat, (
            f"{lv}: fail must include base-loss clause"
        )


def test_two_objective_facts_and_garrison_present():
    """Both objective enemy `fact`s (NE + SE) and a ≤2-rifleman light
    garrison must be present in every level's compiled scenario."""
    p = load_pack(PACK)
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        actors = c.scenario.actors
        enemy_facts = {
            (a.position[0], a.position[1])
            for a in actors
            if a.owner == "enemy" and a.type == "fact"
        }
        assert NE in enemy_facts, f"{lv}: enemy fact at NE {NE} missing"
        assert SE in enemy_facts, f"{lv}: enemy fact at SE {SE} missing"
        enemy_e1s = [a for a in actors if a.owner == "enemy" and a.type == "e1"]
        assert 1 <= len(enemy_e1s) <= 2, (
            f"{lv}: garrison must stay LIGHT (1–2 e1); got {len(enemy_e1s)}"
        )


def test_enemy_is_turtle_bot():
    """`turtle` keeps each garrison at its objective — it does NOT
    chase the agent or harass the base, so the timing teeth are the
    unambiguous discriminator."""
    p = load_pack(PACK)
    assert p.base.get("enemy", {}).get("bot_type") == "turtle"


def test_hard_has_multiple_spawn_point_groups():
    p = load_pack(PACK)
    c = compile_level(p, "hard")
    sp = {
        a.spawn_point if a.spawn_point is not None else 0
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sorted(sp)}"


# ── 2) engine-required scripted-policy sweep ───────────────────────────


def _fact_near(rs, corner, r=9):
    for b in (rs.get("enemy_buildings_summary") or []):
        if (
            str(b.get("type", "")).lower() == "fact"
            and abs(b["cell_x"] - corner[0]) <= r
            and abs(b["cell_y"] - corner[1]) <= r
        ):
            return str(b["id"])
    return None


def _split(rs):
    us = sorted(rs.get("units_summary") or [], key=lambda u: u["cell_y"])
    half = len(us) // 2
    return (
        [str(u["id"]) for u in us[:half]],
        [str(u["id"]) for u in us[half:]],
    )


class _Parallel:
    """Intended play: dispatch BOTH prongs in the same decision turn,
    each to its aligned corner; switch to attack_unit on the fact
    once it is sighted. Should WIN on every level + every seed."""

    def __init__(self) -> None:
        self.n = None
        self.s = None

    def __call__(self, rs, C):
        us = rs.get("units_summary") or []
        if not us:
            return [C.observe()]
        if self.n is None:
            self.n, self.s = _split(rs)
        alive = {str(u["id"]) for u in us}
        n = [i for i in self.n if i in alive]
        s = [i for i in self.s if i in alive]
        cmds = []
        nf = _fact_near(rs, NE)
        if n:
            cmds.append(C.attack_unit(n, nf) if nf else C.attack_move(n, *NE))
        sf = _fact_near(rs, SE)
        if s:
            cmds.append(C.attack_unit(s, sf) if sf else C.attack_move(s, *SE))
        return cmds or [C.observe()]


class _SerialInTime:
    """Anti-pattern: dispatch only the NORTH prong; wait for it to
    raze the NE fact; only THEN dispatch the SOUTH prong. The second
    rush stacks after the first — ~2× make-span. Should LOSE on every
    level + every seed."""

    def __init__(self) -> None:
        self.n = None
        self.s = None
        self.phase = 0
        self.seen_ne = False

    def __call__(self, rs, C):
        us = rs.get("units_summary") or []
        if not us:
            return [C.observe()]
        if self.n is None:
            self.n, self.s = _split(rs)
        alive = {str(u["id"]) for u in us}
        n = [i for i in self.n if i in alive]
        s = [i for i in self.s if i in alive]
        nf = _fact_near(rs, NE)
        sf = _fact_near(rs, SE)
        if self.phase == 0:
            if n and nf:
                self.seen_ne = True
                return [C.attack_unit(n, nf)]
            if n and not nf:
                near = all(
                    abs(u["cell_x"] - NE[0]) <= 10
                    for u in us
                    if str(u["id"]) in n
                )
                if self.seen_ne:
                    self.phase = 1
                else:
                    if near:
                        self.seen_ne = True
                    return [C.attack_move(n, *NE)]
            else:
                self.phase = 1
        if s:
            return [C.attack_unit(s, sf) if sf else C.attack_move(s, *SE)]
        return [C.observe()]


def _stall(rs, C):
    return [C.observe()]


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_parallel_dispatch_wins(lv, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, _Parallel(), seed=seed)
    assert r.outcome == "win", (
        f"{lv}:seed{seed} parallel dispatch must WIN "
        f"(got {r.outcome} at tick {r.signals.game_tick})"
    )


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_serial_loses(lv, seed):
    """A serial play (raze NE, then dispatch the SE prong) stacks the
    two rushes back-to-back and busts the clock — real reachable LOSS
    on every level + every seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, _SerialInTime(), seed=seed)
    assert r.outcome == "loss", (
        f"{lv}:seed{seed} serial play must LOSE on the clock "
        f"(got {r.outcome} at tick {r.signals.game_tick})"
    )


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(lv, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), lv)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{lv}:seed{seed} stall must LOSE (got {r.outcome} "
        f"at tick {r.signals.game_tick})"
    )


def test_hard_seeds_produce_distinct_starts():
    """The two `spawn_point` groups must round-robin under seeds 1..4
    (the contract enforced for `UPGRADED` hard tiers)."""
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
                starts.add(tuple(sorted((x["cell_x"], x["cell_y"]) for x in u)))
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds produced identical starts {starts}; spawn round-robin off"
    )
