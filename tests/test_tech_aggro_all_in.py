"""tech-aggro-all-in pack — full no-cheat validation on Rust.

The Wave-4 AGGRO tech-triple cell: rush highest tech (proc → weap →
tsla) under a tight clock. Pre-placed base = fact + barr + powr; the
AGGRO axis is "skip any economy/defense investment beyond the bare
tech beeline". Cash budget covers the 4600-cost chain with little
slack — medium/hard clocks bite extra structures.

Bar (per CLAUDE.md, "no defect, no cheat"):
  - intended pure-tech-beeline → WIN on every (level, seed).
  - stall (only observe)        → LOSS on every (level, seed).
  - econ-first (extra powr +
    2nd proc before tech)       → LOSS on every (level, seed).
  - balanced (build ftur before
    tsla)                       → LOSS on MEDIUM and HARD; on
                                  easy the looser clock allows it
                                  to win (documented; easy is the
                                  rehearsal tier per the checklist).

Real-world anchors (meta.benchmark_anchor): SC2 all-in cheese tech
rush; startup MVP-or-die runway burn; military rush-to-nuclear.

Engine facts respected (CLAUDE.md):
  - tsla cost=1200 in Defense queue, prereq=weap; weap prereq
    INCLUDES proc; barr is the soviet barracks (tent is allied).
  - within_ticks pairs with `has_building` (no after_ticks-in-win
    footgun); after_ticks is in fail_condition only.
  - hard has 2 agent spawn_point groups (north/south base) so seeds
    round-robin the start position per tests/test_hard_tier.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "tech-aggro-all-in.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _intended_pure_tech_beeline():
    """The intended AGGRO play: proc → weap → tsla, no extras.
    Discovers the fact's cell from the observation to handle the
    seed-round-robined hard spawn (north vs south base)."""
    def pol(obs, Cmd):
        own_b_list = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in own_b_list}
        prod = obs.get("production", []) or []
        fact = next((b for b in own_b_list if b["type"] == "fact"), None)
        if not fact:
            return [Cmd.observe()]
        fx, fy = fact["cell_x"], fact["cell_y"]
        seq = [
            ("proc", fx + 8, fy + 4),
            ("weap", fx + 12, fy),
            ("tsla", fx + 16, fy),
        ]
        cmds = []
        for b, x, y in seq:
            if b not in own_b and b not in prod:
                cmds.append(Cmd.build(b))
                cmds.append(Cmd.place_building(b, x, y))
                break
            elif b not in own_b and b in prod:
                cmds.append(Cmd.place_building(b, x, y))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _econ_first_policy():
    """Builds proc, then a SECOND powr, then a SECOND proc, THEN
    weap + tsla — the classic "secure the economy floor first"
    failure mode. Burns time/queue and misses the tsla deadline."""
    def pol(obs, Cmd):
        own_b_l = obs.get("own_buildings", []) or []
        own_b_types = {b["type"] for b in own_b_l}
        proc_count = sum(1 for b in own_b_l if b["type"] == "proc")
        powr_count = sum(1 for b in own_b_l if b["type"] == "powr")
        prod = obs.get("production", []) or []
        fact = next((b for b in own_b_l if b["type"] == "fact"), None)
        if not fact:
            return [Cmd.observe()]
        fx, fy = fact["cell_x"], fact["cell_y"]
        cmds = []
        if proc_count < 1 and "proc" not in prod:
            cmds.append(Cmd.build("proc"))
            cmds.append(Cmd.place_building("proc", fx + 8, fy + 4))
        elif powr_count < 2 and "powr" not in prod and proc_count >= 1:
            cmds.append(Cmd.build("powr"))
            cmds.append(Cmd.place_building("powr", fx + 4, fy + 8))
        elif proc_count < 2 and "proc" not in prod and powr_count >= 2:
            cmds.append(Cmd.build("proc"))
            cmds.append(Cmd.place_building("proc", fx + 12, fy + 4))
        elif "weap" not in own_b_types and "weap" not in prod and proc_count >= 2:
            cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", fx + 12, fy))
        elif "tsla" not in own_b_types and "tsla" not in prod and "weap" in own_b_types:
            cmds.append(Cmd.build("tsla"))
            cmds.append(Cmd.place_building("tsla", fx + 16, fy))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _balanced_policy():
    """Build the tech chain but ALSO build a defensive turret (ftur)
    in the middle. On easy this still wins (loose clock); on
    medium/hard the extra structure pushes past the deadline."""
    def pol(obs, Cmd):
        own_b_l = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in own_b_l}
        prod = obs.get("production", []) or []
        fact = next((b for b in own_b_l if b["type"] == "fact"), None)
        if not fact:
            return [Cmd.observe()]
        fx, fy = fact["cell_x"], fact["cell_y"]
        seq = [
            ("proc", fx + 8, fy + 4),
            ("ftur", fx + 8, fy - 4),
            ("ftur", fx + 8, fy - 2),
            ("weap", fx + 12, fy),
            ("tsla", fx + 16, fy),
        ]
        cmds = []
        for b, x, y in seq:
            if b in own_b:
                continue
            if b in prod:
                cmds.append(Cmd.place_building(b, x, y))
            else:
                cmds.append(Cmd.build(b))
                cmds.append(Cmd.place_building(b, x, y))
                break
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ─────────────


def test_pack_loads_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "tech-aggro-all-in"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """The Wave-4 spec requires the SC2 cheese-tech / startup MVP /
    rush-to-nuclear anchor list to be present in meta."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2" in a and "tech" in a for a in anchors), anchors
    assert any("startup" in a.lower() for a in anchors), anchors
    assert any("nuclear" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_win_requires_weap_and_tsla_on_every_level():
    """The whole pack is about owning weap AND tsla under the
    clock — confirm the compiled win predicate includes both keys
    via `has_building` leaves (not satisfiable by anything else)."""
    pack = load_pack(PACK)

    def _collect(node, key, out):
        if isinstance(node, dict):
            if key in node:
                out.append(node[key])
            for v in node.values():
                _collect(v, key, out)
        elif isinstance(node, list):
            for v in node:
                _collect(v, key, out)

    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        hbs = []
        _collect(win, "has_building", hbs)
        assert "weap" in hbs, f"{lvl} win missing has_building: weap; got {hbs}"
        assert "tsla" in hbs, f"{lvl} win missing has_building: tsla; got {hbs}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
    pack = load_pack(PACK)

    def _collect(node, key, out):
        if isinstance(node, dict):
            if key in node:
                out.append(node[key])
            for v in node.values():
                _collect(v, key, out)
        elif isinstance(node, list):
            for v in node:
                _collect(v, key, out)

    for lvl in LEVELS:
        max_turns = pack.levels[lvl].max_turns
        reachable = 93 + 90 * (max_turns - 1)
        wts = []
        _collect(
            compile_level(pack, lvl).win_condition.model_dump(exclude_none=True),
            "within_ticks", wts,
        )
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


def test_hard_has_multiple_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (UPGRADED
    contract — tests/test_hard_tier.py also enforces it once the
    pack is added to the UPGRADED list)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_hard_has_building_total_cap_predicate():
    """The hard tier's `building_total_gte: 6` is the load-bearing
    knob that locks out a buffer-build (the agent must build
    EXACTLY the 6-building bare chain). Confirm it's in the win."""
    win = compile_level(load_pack(PACK), "hard").win_condition.model_dump(exclude_none=True)

    def _has_key(node, k):
        if isinstance(node, dict):
            if k in node:
                return True
            return any(_has_key(v, k) for v in node.values())
        if isinstance(node, list):
            return any(_has_key(v, k) for v in node)
        return False

    assert _has_key(win, "building_total_gte"), (
        f"hard win missing building_total_gte cap; got {win}"
    )


# ── Engine-bound tests (parameterised over seeds 1..4) ───────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_pure_tech_beeline_wins(level, seed):
    """The intended AGGRO play — proc → weap → tsla, no extras —
    must WIN on every (level, seed). Load-bearing solvability
    test: the pack is winnable inside the budget by the advertised
    capability (skip-economy tech rush)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_pure_tech_beeline(), seed=seed)
    assert res.outcome == "win", (
        f"intended pure-tech-beeline must WIN on {level} s={seed}; "
        f"got {res.outcome} t={res.turns} tick={res.signals.game_tick} "
        f"own={sorted(res.signals.own_building_types)}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_econ_first_loses(level, seed):
    """Econ-first (extra powr + 2nd proc before tech) must LOSE on
    every (level, seed) — the extra economy spend never finishes
    the tsla before the deadline (even on the looser easy clock,
    queue serialisation + cash drain blow past 4500 ticks)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _econ_first_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"econ-first must LOSE on {level} s={seed}; got {res.outcome} "
        f"t={res.turns} own={sorted(res.signals.own_building_types)}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", ("medium", "hard"))
def test_balanced_loses_on_medium_and_hard(level, seed):
    """Balanced (proc + ftur + ftur + weap + tsla) must LOSE on
    medium and hard — the extra defensive structures serialise the
    queue and miss the tsla deadline.

    EASY is excluded by design: the loose clock (4500 ticks) is
    intentionally generous enough for balanced to still win — easy
    is the rehearsal tier (the AGGRO axis only bites under the
    medium/hard clock). This is the documented escalation: easy
    teaches the idiom, medium/hard discriminate it."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _balanced_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"balanced must LOSE on {level} s={seed}; got {res.outcome} "
        f"t={res.turns} own={sorted(res.signals.own_building_types)}"
    )
