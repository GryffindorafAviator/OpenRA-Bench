"""tp-decision-under-clock — quick decision under a tight clock.

Two enemy construction yards sit at opposite far corners (NE (116,6),
SE (116,34)); one is lightly defended (2× e1 picket), the other
heavily (3× 3tnk + 2× e3). A 4× 2tnk column + 2 scout jeeps stage at
the west base. The capability under test is a brief, bounded scout
(jeeps read which corner is soft) followed by an irreversible commit
of the slow column to the EASIER target — raze its `fact` before the
clock runs out with ≥2 own units alive.

Hard rotates the light/heavy assignment per ENEMY `spawn_point` group
(Wave-9 per-owner spawn filter): spawn 0 → NE light / SE heavy;
spawn 1 → NE heavy / SE light. A fixed-corner opener wins half the
hard seeds and loses the other half.

Scripted policies pin the "no defect, no cheat" bar:

  * stall                 → LOSS (clock; nothing dispatched)
  * commit-blind (fixed)  → WIN on the seed its corner is light,
                            LOSS on the seed its corner is heavy
                            (column wiped below own_units_gte:2)
  * dither / over-scout   → LOSS (column held at base too long;
                            the late commit busts the clock)
  * intended brief-scout  → WIN every level + every hard seed
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK_PATH = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "tp-decision-under-clock.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)
NE = (72, 6)
SE = (72, 34)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "tp-decision-under-clock"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: anchors must call out the poker-speed /
    ops-triage / decision-under-uncertainty framing."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("poker speed decision" in a for a in anchors), anchors
    assert any("tense ops triage" in a for a in anchors), anchors
    assert any("decision under uncertainty" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout,
    base-collapse, or column-wipe."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / after_ticks must be reachable inside max_turns —
    engine advances ~90 ticks/turn → reachable = 93 + 90·(N-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        comp = compile_level(pack, lvl)
        wts: list[int] = []
        afs: list[int] = []

        def _collect(node):
            if isinstance(node, dict):
                if "within_ticks" in node:
                    wts.append(node["within_ticks"])
                if "after_ticks" in node:
                    afs.append(node["after_ticks"])
                for v in node.values():
                    _collect(v)
            elif isinstance(node, list):
                for v in node:
                    _collect(v)

        _collect(comp.win_condition.model_dump(exclude_none=True))
        _collect(comp.fail_condition.model_dump(exclude_none=True))
        assert wts, f"{lvl} has no within_ticks leaf"
        assert afs, f"{lvl} has no after_ticks leaf"
        for k in wts + afs:
            assert k <= reachable, (
                f"{lvl}: deadline {k} > reachable {reachable} "
                f"(max_turns={level_def.max_turns}) — never bites ⇒ draw"
            )


def test_win_requires_either_corner_fact_destroyed():
    """Win must be an any_of over the NE and SE corner facts — either
    enemy construction yard razed satisfies the objective."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        regions: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "enemy_key_buildings_destroyed_in_region" in node:
                    regions.append(node["enemy_key_buildings_destroyed_in_region"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        ys = sorted(int(r["y"]) for r in regions)
        assert ys == [NE[1], SE[1]], (
            f"{lvl}: win must target both corner facts (y={NE[1]} and "
            f"y={SE[1]}), got region ys {ys}"
        )
        for r in regions:
            assert "fact" in [str(t).lower() for t in r["types"]]


def test_win_has_own_units_floor():
    """own_units_gte:2 is the anti-commit-blind teeth — charging the
    heavy garrison wipes the column and fails the win."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        found: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "own_units_gte" in node:
                    found.append(node["own_units_gte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert found and all(int(n) >= 2 for n in found), (
            f"{lvl}: win must require own_units_gte ≥ 2, got {found}"
        )


def test_each_level_has_column_jeeps_and_two_enemy_facts():
    """Every tier must field the 4× 2tnk column + 2 scout jeeps + an
    agent fact, and TWO enemy facts (the two-option decision)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent = [a for a in c.scenario.actors if a.owner == "agent"]
        enemy = [a for a in c.scenario.actors if a.owner == "enemy"]
        n_tnk = sum(1 for a in agent if str(a.type).lower() == "2tnk")
        n_jeep = sum(1 for a in agent if str(a.type).lower() == "jeep")
        assert n_tnk == 4, f"{lvl}: expected 4 2tnk, got {n_tnk}"
        assert n_jeep == 2, f"{lvl}: expected 2 jeep, got {n_jeep}"
        assert any(str(a.type).lower() == "fact" for a in agent), f"{lvl}: no agent fact"
        # hard duplicates facts across 2 spawn groups → 4 enemy facts;
        # easy/medium have exactly 2.
        n_efact = sum(1 for a in enemy if str(a.type).lower() == "fact")
        assert n_efact >= 2, f"{lvl}: expected ≥2 enemy facts, got {n_efact}"


def test_hard_has_two_enemy_spawn_groups():
    """Hard's seed axis is enemy-side: ≥2 distinct enemy spawn_point
    groups so the light/heavy assignment flips per seed."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "enemy"}
    assert len(sp) >= 2, f"hard needs ≥2 enemy spawn groups, got {sp}"
    # the agent base must NOT declare spawn_point (fixed every seed).
    asp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert asp == {None}, f"agent base must be fixed (no spawn_point), got {asp}"


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _fact_id_if_close(rs, corner, tanks):
    """The corner fact's id — but only once a tank is within 12 cells
    of it. Approaching with attack_move and only focus-firing
    (attack_unit) at close range is the realistic intended pattern."""
    for e in rs.get("enemy_summary", []) or []:
        if (
            e.get("type") == "fact"
            and abs(e["cell_x"] - corner[0]) < 8
            and abs(e["cell_y"] - corner[1]) < 8
        ):
            for u in tanks:
                if (
                    abs(u["cell_x"] - e["cell_x"]) < 12
                    and abs(u["cell_y"] - e["cell_y"]) < 12
                ):
                    return str(e["id"])
    return None


def _make_commit_blind(corner):
    """Send the column straight at a FIXED corner and focus-fire its
    fact. Wins if the corner is light, loses (column wiped) if heavy."""

    def pol(rs, Command):
        tanks = [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]
        if not tanks:
            return [Command.observe()]
        ids = [str(u["id"]) for u in tanks]
        fid = _fact_id_if_close(rs, corner, tanks)
        if fid:
            return [Command.attack_unit(ids, fid)]
        return [Command.attack_move(ids, target_x=corner[0], target_y=corner[1])]

    return pol


def _make_dither(corner):
    """Over-scout: jeeps scout, but the column is HELD at base until
    t≈2300 (waiting for full certainty) — the late commit busts the
    tight clock even when it targets the light corner."""

    def pol(rs, Command):
        t = rs.get("game_tick", 0)
        us = rs.get("units_summary", [])
        jeeps = [u for u in us if u.get("type") == "jeep"]
        tanks = [u for u in us if u.get("type") == "2tnk"]
        if not tanks:
            return [Command.observe()]
        ids = [str(u["id"]) for u in tanks]
        if t < 50:
            cmds = []
            if len(jeeps) >= 1:
                cmds.append(
                    Command.move_units(
                        [str(jeeps[0]["id"])], target_x=70, target_y=NE[1]
                    )
                )
            if len(jeeps) >= 2:
                cmds.append(
                    Command.move_units(
                        [str(jeeps[1]["id"])], target_x=70, target_y=SE[1]
                    )
                )
            return cmds or [Command.observe()]
        if t < 2300:
            return [Command.observe()]
        fid = _fact_id_if_close(rs, corner, tanks)
        if fid:
            return [Command.attack_unit(ids, fid)]
        return [Command.attack_move(ids, target_x=corner[0], target_y=corner[1])]

    return pol


def _make_intended():
    """Brief-scout-then-commit: roll the column out toward centre at
    t=0 while the jeeps race to the two corners. The FIRST jeep lost
    marks the HEAVY corner; the surviving jeep (or its last-known
    position) marks the LIGHT corner. Divert the column to the light
    corner and focus-fire that fact once a tank is close."""
    st = {"corner": None, "njeeps": 2, "last_jeep_y": {}}

    def pol(rs, Command):
        t = rs.get("game_tick", 0)
        us = rs.get("units_summary", [])
        jeeps = [u for u in us if u.get("type") == "jeep"]
        # Track which jeep ids we've seen and where they last were.
        for j in jeeps:
            st["last_jeep_y"][str(j["id"])] = (
                int(j["cell_y"]), int(j["cell_x"])
            )
        tanks = [u for u in us if u.get("type") == "2tnk"]
        if len(jeeps) < st["njeeps"]:
            st["njeeps"] = len(jeeps)
        if not tanks:
            return [Command.observe()]
        ids = [str(u["id"]) for u in tanks]
        if t < 50:
            cmds = [Command.attack_move(ids, target_x=70, target_y=20)]
            if len(jeeps) >= 1:
                cmds.append(
                    Command.move_units(
                        [str(jeeps[0]["id"])], target_x=70, target_y=NE[1]
                    )
                )
            if len(jeeps) >= 2:
                cmds.append(
                    Command.move_units(
                        [str(jeeps[1]["id"])], target_x=70, target_y=SE[1]
                    )
                )
            return cmds
        if st["corner"] is None:
            if st["njeeps"] < 2 and jeeps:
                # One jeep died, one survives. Surviving jeep is in
                # the LIGHT corner — commit there.
                st["corner"] = NE if jeeps[0]["cell_y"] < 20 else SE
            elif st["njeeps"] == 0:
                # Both jeeps died before reporting back. Pick the
                # corner where the LATER-dying jeep got further east
                # (it lived longer = lighter resistance). Falls back
                # to NE if last positions are equivalent.
                ys = [v[0] for v in st["last_jeep_y"].values()]
                if ys:
                    # Heuristic: pick the y closer to the FURTHER jeep's
                    # last seen position. With only y info, default to
                    # the corner closest to the average last-y.
                    avg_y = sum(ys) / len(ys)
                    st["corner"] = NE if avg_y < 20 else SE
                else:
                    st["corner"] = NE
            else:
                return [Command.attack_move(ids, target_x=70, target_y=20)]
        c = st["corner"]
        fid = _fact_id_if_close(rs, c, tanks)
        if fid:
            return [Command.attack_unit(ids, fid)]
        return [Command.attack_move(ids, target_x=c[0], target_y=c[1])]

    return pol


# ── Solvency: intended brief-scout-then-commit WINS every (lvl,seed) ──


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_brief_scout_then_commit_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _make_intended(), seed=seed)
    assert res.outcome == "win", (
        f"intended brief-scout-then-commit must WIN on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick}"
    )


# ── Stability: every non-capability play is a real reachable LOSS ──


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
@pytest.mark.parametrize("corner", [NE, SE])
def test_dither_overscout_loses(level, seed, corner):
    """Holding the column at base for the full scout (over-scout)
    busts the clock on every tier, even when it commits to the light
    corner — a real reachable clock LOSS, never a draw."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _make_dither(corner), seed=seed)
    assert res.outcome == "loss", (
        f"dither/over-scout (corner={corner}) must LOSE on {level} "
        f"s={seed}; got {res.outcome} tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_commit_blind_fixed_corner_is_structural_loss_on_hard(seed):
    """A pre-committed fixed corner WINS the hard seed its corner is
    light and LOSES the hard seed its corner is heavy — so neither
    fixed corner wins every seed. The hard-tier contract (win on
    every seed 1..4) makes any blind opener a structural LOSS."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    ne = run_level(c, _make_commit_blind(NE), seed=seed)
    se = run_level(c, _make_commit_blind(SE), seed=seed)
    # exactly one of the two fixed corners wins this seed (they flip).
    outcomes = {ne.outcome, se.outcome}
    assert outcomes == {"win", "loss"}, (
        f"hard s={seed}: a fixed corner must win XOR lose (light/heavy "
        f"flip); got NE={ne.outcome} SE={se.outcome}"
    )


def test_commit_blind_loses_on_at_least_one_hard_seed_each_corner():
    """Across the 4 hard seeds, each fixed-corner opener must LOSE at
    least once — proving no memorised corner generalises."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    for corner, name in ((NE, "NE"), (SE, "SE")):
        outs = [
            run_level(c, _make_commit_blind(corner), seed=s).outcome for s in SEEDS
        ]
        assert "loss" in outs, (
            f"commit-blind-{name} must LOSE on ≥1 hard seed; got {outs}"
        )
        assert "win" in outs, (
            f"commit-blind-{name} must WIN on ≥1 hard seed; got {outs}"
        )
