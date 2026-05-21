"""mfb-mirror-base-east-west — geographic mirror redundancy under a
coordinated attack. Wave-9 REASONING pack.

The agent owns a fully built WEST base (fact + proc + powr + HoldFire
medium tanks) and a HALF-built EAST standby (just the fact + HoldFire
tanks). A coordinated grenadier probe hits BOTH bases. The intended
play MIRRORS the east standby to parity (build powr → proc inside the
east region) WHILE commanding BOTH HoldFire defender groups onto their
on-latitude attack band — the tanks never engage on their own.

The bar (CLAUDE.md "no defect, no cheat"), every level + seed 1..4:

  * stall            → LOSS: HoldFire tanks never fire, both rush
                       bands raze both facts.
  * defend-west-only → LOSS: the EAST fact is razed (its defenders
                       sit idle, no east mirror) — fact pair clause
                       fails.
  * defend-east-only → LOSS: symmetric — the WEST fact falls.
  * build-only       → LOSS: both rushes converge unopposed; both
                       facts fall before the deadline.
  * intended         → WIN: command both defender groups AND stream
                       the east powr→proc; both replicas survive and
                       a proc lands inside each region.

Recalibration note: the engine combat rebalance (armor-class weapon
selection) made grenadiers markedly stronger versus medium tanks, so
the per-base defender count was raised (easy 2→3, medium/hard 2→4)
to keep the commanded-defence path winning while idle / single-side
play still loses. The capability stays load-bearing — a stall or a
one-side-only defence still loses every level/seed.
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
    / "mfb-mirror-base-east-west.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "mfb-mirror-base-east-west"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_benchmark_anchor_declared():
    pack = load_pack(PACK_PATH)
    anchors = [a.lower() for a in (pack.meta.benchmark_anchor or [])]
    assert anchors
    assert any("mirror" in a or "replica" in a or "multi-region" in a
               for a in anchors), anchors


def test_every_level_has_fail_condition():
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.win_condition is not None, f"{lvl} missing win_condition"
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / fail.after_ticks must be reachable inside
    max_turns (engine ~90 ticks/turn → reachable = 93 + 90·(N-1)) so a
    non-win run is a real LOSS, not a silent DRAW."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True
        )
        fail = compile_level(pack, lvl).fail_condition.model_dump(
            exclude_none=True
        )

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
        assert wts and all(wt <= reachable for wt in wts), (
            f"{lvl} within_ticks unreachable: {wts} > {reachable}"
        )
        ats: list[int] = []
        _collect(fail, "after_ticks", ats)
        assert ats and max(ats) <= reachable, (
            f"{lvl} fail.after_ticks unreachable: {ats} > {reachable}"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (UPGRADED
    contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    sp.discard(None)
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


# ── Scripted-policy helpers ───────────────────────────────────────


def _bld(rs):
    return rs.get("_raw", {}).get("own_buildings", []) or []


def _facts(rs):
    return [b for b in _bld(rs) if b["type"] == "fact"]


def _tanks(rs):
    return [u for u in (rs.get("units_summary") or [])
            if str(u.get("type", "")).lower() == "2tnk"]


def _enemies(rs):
    en = rs.get("_raw", {}).get("enemy_positions") or []
    # exclude the far-corner unarmed anti-auto-done marker
    return [e for e in en if not (e["cell_x"] > 120 and e["cell_y"] > 40)]


def _midline(rs):
    facts = _facts(rs)
    if len(facts) >= 2:
        xs = sorted(f["cell_x"] for f in facts)
        return (xs[0] + xs[-1]) / 2
    return 50.0


def _east_coords(rs):
    facts = _facts(rs)
    if facts:
        return max(f["cell_x"] for f in facts), facts[0]["cell_y"]
    return 85, 20


def _command_side(C, tanks, enemies, fallback_x, y):
    if not tanks:
        return None
    tids = [str(u["id"]) for u in tanks]
    if enemies:
        return C.attack_unit(tids, str(enemies[0]["id"]))
    return C.attack_move(tids, target_x=fallback_x, target_y=y)


def _stall(_rs, C):
    return [C.observe()]


def _defend_one(side):
    """Command only ONE base's tanks; never build, never touch the
    other base."""
    def pol(rs, C):
        mid = _midline(rs)
        tanks = _tanks(rs)
        en = _enemies(rs)
        facts = _facts(rs)
        y = facts[0]["cell_y"] if facts else 20
        if side == "west":
            t = [u for u in tanks if u["cell_x"] < mid]
            e = [x for x in en if x["cell_x"] < mid]
            fb = min((f["cell_x"] for f in facts), default=15) + 12
        else:
            t = [u for u in tanks if u["cell_x"] >= mid]
            e = [x for x in en if x["cell_x"] >= mid]
            fb = max((f["cell_x"] for f in facts), default=85) - 12
        cmd = _command_side(C, t, e, fb, y)
        return [cmd] if cmd else [C.observe()]
    return pol


def _build_only():
    """Mirror the east (powr → proc) but never command any defender."""
    st = {"phase": 0}

    def pol(rs, C):
        ex, y = _east_coords(rs)
        bld = _bld(rs)
        cmds = []
        if st["phase"] == 0:
            cmds.append(C.build("powr"))
            st["phase"] = 1
        elif st["phase"] == 1:
            cmds.append(C.place_building("powr", ex - 3, y))
            if any(b["type"] == "powr" and abs(b["cell_x"] - (ex - 3)) <= 2
                   for b in bld):
                st["phase"] = 2
        elif st["phase"] == 2:
            cmds.append(C.build("proc"))
            st["phase"] = 3
        elif st["phase"] == 3:
            cmds.append(C.place_building("proc", ex + 3, y))
        return cmds or [C.observe()]
    return pol


def _intended():
    """Mirror-both: stream the east powr→proc AND command BOTH
    HoldFire defender groups onto their on-side attack band."""
    st = {"phase": 0}

    def pol(rs, C):
        mid = _midline(rs)
        tanks = _tanks(rs)
        en = _enemies(rs)
        facts = _facts(rs)
        y = facts[0]["cell_y"] if facts else 20
        xs = sorted(f["cell_x"] for f in facts) if facts else [15, 85]
        wx, ex = xs[0], xs[-1]
        cmds = []
        cw = _command_side(
            C,
            [u for u in tanks if u["cell_x"] < mid],
            [x for x in en if x["cell_x"] < mid],
            wx + 12, y,
        )
        ce = _command_side(
            C,
            [u for u in tanks if u["cell_x"] >= mid],
            [x for x in en if x["cell_x"] >= mid],
            ex - 12, y,
        )
        if cw:
            cmds.append(cw)
        if ce:
            cmds.append(ce)
        bld = _bld(rs)
        if st["phase"] == 0:
            cmds.append(C.build("powr"))
            st["phase"] = 1
        elif st["phase"] == 1:
            cmds.append(C.place_building("powr", ex - 3, y))
            if any(b["type"] == "powr" and abs(b["cell_x"] - (ex - 3)) <= 2
                   for b in bld):
                st["phase"] = 2
        elif st["phase"] == 2:
            cmds.append(C.build("proc"))
            st["phase"] = 3
        elif st["phase"] == 3:
            cmds.append(C.place_building("proc", ex + 3, y))
        return cmds or [C.observe()]
    return pol


# ── The bar — every lazy / one-side play LOSES, intended WINS ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE, got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_defend_west_only_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _defend_one("west"), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: defend-west-only must LOSE "
        f"(EAST fact razed), got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_defend_east_only_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _defend_one("east"), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: defend-east-only must LOSE "
        f"(WEST fact razed), got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_only_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _build_only(), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: build-only (no defence) must LOSE "
        f"(both facts razed), got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_mirror_both_wins(level, seed):
    """Command both defender groups AND stream the east mirror — both
    replicas survive and a proc lands inside each region → WIN."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended(), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed{seed}: intended mirror-both must WIN, "
        f"got {res.outcome}"
    )
