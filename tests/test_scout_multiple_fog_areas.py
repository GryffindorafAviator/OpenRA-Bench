"""scout-multiple-fog-areas — parallel multi-region scout pack.

Capability: PERCEPTION (parallel dispatch under non-overlapping
coverage / multi-scout intel-window). Cell of the perception ladder
sibling to perception-frontier-reading (single-frontier read) and
perception-count-the-threat (exact census under fog).

Scripted policies cover the bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall              → LOSS (clock; nothing discovered)
  * single-jeep tour   → LOSS on medium+hard (clock; one jeep
                         cannot cover K regions in time)
  * all-to-NE-only     → LOSS (K-1 region buildings stay in fog,
                         win bar buildings_discovered_gte:K
                         is never met)
  * intended SPLIT     → WIN (every level, seeds 1..4)

Easy is intentionally permissive: K=2 makes a single-jeep tour
(NE→SE on the east edge, ~166 cells ≈ 1000 ticks) fit inside the
generous 1800-tick clock. The discrimination teeth bite on medium
and hard, where the tour distance and K-bar combine to force
genuine parallel dispatch.

Hard tier rotates the agent spawn corner between NW (5, 6..12) and
SW (5, 28..34) by seed (UPGRADED contract). The intended split
policy reads the actual jeep rows on turn 1 and chooses the north
sweep (NW→NE on y≈4) or south sweep (SW→SE on y≈36) per jeep based
on which corridor each jeep is on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK_PATH = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "scout-multiple-fog-areas.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-multiple-fog-areas"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: the anchors must call out the
    Watch-And-Help / SMAC / SC2 multi-scout / military distributed
    recon framing."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("Watch-And-Help" in a for a in anchors), anchors
    assert any("SMAC" in a for a in anchors), anchors
    assert any("SC2 multi-scout" in a for a in anchors), anchors
    assert any("military distributed reconnaissance" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout
    or scout-death."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (the binding
    contract from tests/test_hard_tier.py::UPGRADED)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / after_ticks must be reachable inside max_turns.
    Engine advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        fail = compile_level(pack, lvl).fail_condition.model_dump(exclude_none=True)

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list[int] = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
        ats: list[int] = []
        _collect(fail, "after_ticks", ats)
        for at in ats:
            assert at <= reachable, (
                f"{lvl} fail.after_ticks={at} > reachable={reachable} "
                f"(max_turns={max_turns}) — fail clause never bites ⇒ draw"
            )


def test_win_uses_buildings_discovered_gte():
    """Capability contract: the win predicate must enforce
    multi-region discovery via buildings_discovered_gte (not a
    laxer landmark/region predicate that a single-region rush
    could satisfy)."""
    pack = load_pack(PACK_PATH)
    expected_k = {"easy": 2, "medium": 3, "hard": 4}
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        ks: list[int] = []

        def _walk(node):
            if isinstance(node, dict):
                if "buildings_discovered_gte" in node:
                    ks.append(node["buildings_discovered_gte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert ks == [expected_k[lvl]], (
            f"{lvl}: expected buildings_discovered_gte={expected_k[lvl]}, "
            f"got {ks}"
        )


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _all_to_ne(rs, Command):
    """Send the entire fleet to the NE corner — the "send all jeeps to
    one region" failure mode. Discovers at most one region marker;
    leaves the others in fog so the win bar
    buildings_discovered_gte:K (K≥2) is never met."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=124, target_y=4)
        for u in units
    ]


def _single_jeep_tour_factory(targets):
    """Designate the lowest-id jeep as the lone scout and march it
    through `targets` in order. The other jeeps sit at spawn (no
    command issued for them). Models the "one column does all the
    work" failure mode that the parallel-dispatch tooth catches."""
    state = {"phase": 0}

    def _tour(rs, Command):
        units = rs.get("units_summary", []) or []
        if not units:
            return [Command.observe()]
        u = sorted(units, key=lambda x: x["id"])[0]
        uid = str(u["id"])
        ux, uy = u["cell_x"], u["cell_y"]
        if state["phase"] >= len(targets):
            return [Command.observe()]
        tx, ty = targets[state["phase"]]
        if abs(ux - tx) <= 5 and abs(uy - ty) <= 5:
            state["phase"] += 1
            if state["phase"] >= len(targets):
                return [Command.observe()]
            tx, ty = targets[state["phase"]]
        return [Command.move_units([uid], target_x=tx, target_y=ty)]

    return _tour


# Per-level single-jeep tour routes — straight-line east-edge tours
# (the natural "lazy" tour a model would pick).
_TOUR_TARGETS = {
    "easy":   [(124, 4), (124, 36)],
    "medium": [(124, 4), (124, 20), (124, 36)],
    "hard":   [(60, 4), (124, 4), (124, 36), (60, 36)],
}


def _intended_split_factory(level: str):
    """Parallel dispatch: one jeep per region (or one per corridor
    sweep on hard). Reads the actual jeep rows on turn 1 and assigns
    the closest region to each jeep — works for both NW and SW spawn
    on hard.

    easy   (K=2): top jeep → NE (124,4); bottom jeep → SE (124,36);
                  mid jeep idle (only 2 regions).
    medium (K=3): top jeep → NE; mid jeep → far-E (124,20);
                  bottom jeep → SE.
    hard   (K=4): top jeep sweeps north corridor (60,4)→(124,4);
                  bottom jeep sweeps south corridor (60,36)→(124,36);
                  mid jeep idle (3 jeeps for 4 regions ⇒ sweep both
                  ends of one corridor with one jeep). State per
                  sweep jeep is latched so each marker is committed
                  to before moving to the next.
    """
    state = {"top_phase": 0, "bot_phase": 0}

    def _split(rs, Command):
        units = rs.get("units_summary", []) or []
        if not units:
            return [Command.observe()]
        sorted_units = sorted(units, key=lambda u: u["cell_y"])
        cmds = []
        if level == "easy":
            # 3 jeeps, 2 regions — top→NE, bottom→SE, mid idle.
            cmds.append(Command.move_units(
                [str(sorted_units[0]["id"])], target_x=124, target_y=4))
            cmds.append(Command.move_units(
                [str(sorted_units[-1]["id"])], target_x=124, target_y=36))
        elif level == "medium":
            # 3 jeeps, 3 regions — one each.
            targets = [(124, 4), (124, 20), (124, 36)]
            for u, (tx, ty) in zip(sorted_units, targets):
                cmds.append(Command.move_units(
                    [str(u["id"])], target_x=tx, target_y=ty))
        else:  # hard — 3 jeeps, 4 regions, two sweeps.
            top = sorted_units[0]
            bot = sorted_units[-1]
            n_targets = [(60, 4), (124, 4)]
            if state["top_phase"] < len(n_targets):
                tx, ty = n_targets[state["top_phase"]]
                ux, uy = top["cell_x"], top["cell_y"]
                if abs(ux - tx) <= 4 and abs(uy - ty) <= 4:
                    state["top_phase"] += 1
                if state["top_phase"] < len(n_targets):
                    tx, ty = n_targets[state["top_phase"]]
                    cmds.append(Command.move_units(
                        [str(top["id"])], target_x=tx, target_y=ty))
            s_targets = [(60, 36), (124, 36)]
            if state["bot_phase"] < len(s_targets):
                tx, ty = s_targets[state["bot_phase"]]
                ux, uy = bot["cell_x"], bot["cell_y"]
                if abs(ux - tx) <= 4 and abs(uy - ty) <= 4:
                    state["bot_phase"] += 1
                if state["bot_phase"] < len(s_targets):
                    tx, ty = s_targets[state["bot_phase"]]
                    cmds.append(Command.move_units(
                        [str(bot["id"])], target_x=tx, target_y=ty))
        return cmds or [Command.observe()]

    return _split


# ── Solvency: intended SPLIT WINS every (level, seed) ─────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_split_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended_split_factory(level), seed=seed)
    assert res.outcome == "win", (
        f"intended split must WIN on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)} "
        f"lost={res.signals.units_lost}"
    )


# ── Stability: every non-win pattern is a real reachable LOSS ─────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_to_ne_only_loses(level, seed):
    """Sending the whole fleet to one corner discovers at most 1 of
    K region buildings — the win bar (K≥2) is never met and the
    other markers stay in fog (so the engine does NOT auto-`done`),
    meaning the deadline cleanly fires the LOSS."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _all_to_ne, seed=seed)
    assert res.outcome == "loss", (
        f"all-to-NE must LOSE on {level} s={seed}; got {res.outcome} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)} "
        f"lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", ["medium", "hard"])
def test_single_jeep_tour_loses_medium_and_hard(level, seed):
    """One jeep walks A→B→C(→D) serially while the other two sit
    at spawn. The tour distance pushes the discovery of the last
    region past the within_ticks deadline ⇒ LOSS on the clock.

    Easy (K=2 along the east edge) is intentionally permissive: the
    tour distance ≈ 166 cells fits the 1800-tick generous clock,
    so a single-jeep tour CAN win on easy. The bar is medium/hard
    where the K-bar and clock combine to force parallel dispatch."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _single_jeep_tour_factory(_TOUR_TARGETS[level]), seed=seed)
    assert res.outcome == "loss", (
        f"single-jeep tour must LOSE on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}"
    )


# ── Spawn variation contract (hard) ───────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_outcomes_per_seed(seed):
    """Hard's two spawn_point groups round-robin per seed; the
    intended split policy must WIN from EITHER spawn corner. The
    cross-seed distinct-starts assertion is enforced centrally in
    tests/test_hard_tier.py::test_curated_hard_still_compiles_and_runs."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, _intended_split_factory("hard"), seed=seed)
    assert res.outcome == "win", (
        f"hard intended split must WIN s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}"
    )
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
