"""lh-defense-tech-second-base — full no-cheat validation on Rust.

Wave-5 Group G long-horizon "secure-expand-with-tech" pack. Uses the
Wave-2 `then:` happened-before composite to enforce strict 3- (or 4-)
phase ordering:
  PHASE 1: building_count_gte:{type:pbox, n:2|3|3}    # defend
  PHASE 2: has_building: weap                          # tech
  PHASE 3: building_in_region:{x:130,y:30,r:8,
                              type:fact,count:1}      # 2nd base
  PHASE 4 (hard only):
           building_count_gte:{type:proc, n:2}         # second refinery

Bar (per CLAUDE.md): the intended secure-expand-with-tech policy must
WIN on every (level, seed); stall / pure-expand-skip-defence /
pure-defence-no-expand must LOSE on every (level, seed). No draws.

Scenario shape:
  - 160x60 generator-spec arena, allied agent, soviet `patrol`-bot.
  - Pre-placed base: fact + tent + 2× powr + proc + harv + mine
    + 3 rifleman defenders (so income flows from turn 1 and own_units_
    gte:1 is satisfied immediately).
  - Spare MCV staged ~30 cells east of base #1 — agent drives it
    further east and deploys at (131,31) → fact lands at (130,30),
    the centre of the east target region.
  - Inert enemy `fact` marker at (154,4) prevents auto-DRAW.
  - hard: ≥2 spawn_point groups (NORTH y=22 / SOUTH y=38) per the
    hard-tier contract.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "lh-defense-tech-second-base.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; do not run the engine) ─────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "lh-defense-tech-second-base"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    joined = " ".join(anchors).lower()
    assert "sc2" in joined and "secure-expand" in joined, anchors
    assert "planbench" in joined, anchors
    assert "roadmap" in joined or "harden" in joined, anchors
    assert "industrial" in joined or "second plant" in joined, anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_state_based_win_predicate():
    """v1.0 sweep audit (F8 long-horizon): the win is now STATE-BASED
    `all_of:` — pbox≥N + weap + 2nd-fact-in-region (+ proc≥2 on hard)
    + base-alive + within_ticks. The strict `then:` has been removed;
    the patrol-pressure + clock budget still force the secure-expand
    tempo (stall loses to the patrol; pure-expand loses base #1)."""
    pbox_n = {"easy": 2, "medium": 3, "hard": 3}
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK_PATH), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        assert not any("then" in clause for clause in ao), (
            f"{lvl} should be state-based, found `then:` in {win}"
        )
        # Required state clauses present.
        bc = [
            cl["building_count_gte"]
            for cl in ao if "building_count_gte" in cl
        ]
        pbox_clause = next(
            (b for b in bc if b.get("type") == "pbox"), None
        )
        assert pbox_clause is not None and pbox_clause["n"] == pbox_n[lvl], (
            f"{lvl} missing pbox state clause with n={pbox_n[lvl]}: {win}"
        )
        # weap as has_building OR fact-alive as has_building — either way
        # there must be at least 2 has_building clauses (weap + fact-base).
        hb = [cl["has_building"] for cl in ao if "has_building" in cl]
        assert "weap" in hb, f"{lvl} missing has_building:weap clause"
        assert "fact" in hb, f"{lvl} missing has_building:fact (base alive)"
        # 2nd-base region clause.
        br = next(
            (cl["building_in_region"] for cl in ao
             if "building_in_region" in cl), None
        )
        assert br is not None and br["x"] == 130 and br["y"] == 30
        # Hard adds proc≥2.
        if lvl == "hard":
            proc_clause = next(
                (b for b in bc if b.get("type") == "proc"), None
            )
            assert proc_clause is not None and proc_clause["n"] == 2, (
                f"hard missing proc≥2 state clause: {win}"
            )


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns (~90 ticks/turn)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)
        wts: list = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


def _ctx(own_buildings=(), tick=1000):
    """Synthesize a WinContext from a building list."""
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        own_buildings=list(own_buildings),
        own_building_types={t for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        cash=0,
        then_progress={},
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # WIN: full chain — pbox×2 → weap → 2nd fact in east region
    own = [
        ("fact", 10, 30),
        ("pbox", 12, 30), ("pbox", 12, 31),
        ("weap", 14, 30),
        ("fact", 130, 30),
    ]
    assert evaluate(c.win_condition, _ctx(own, tick=3000))
    # FAIL: skip pbox — chain never advances past phase 1.
    own_no_def = [("fact", 10, 30), ("weap", 14, 30), ("fact", 130, 30)]
    assert not evaluate(c.win_condition, _ctx(own_no_def, tick=3000))
    # FAIL: skip expand — chain stuck at phase 3.
    own_no_exp = [
        ("fact", 10, 30),
        ("pbox", 12, 30), ("pbox", 12, 31),
        ("weap", 14, 30),
    ]
    assert not evaluate(c.win_condition, _ctx(own_no_exp, tick=3000))
    # FAIL: no fact at all (base #1 lost).
    assert evaluate(c.fail_condition, _ctx([], tick=3000))
    # FAIL: past deadline.
    assert evaluate(c.fail_condition, _ctx(own, tick=8100))


def test_predicates_medium():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # WIN: pbox×3 → weap → 2nd fact in east region
    own = [
        ("fact", 10, 30),
        ("pbox", 12, 30), ("pbox", 12, 31), ("pbox", 12, 32),
        ("weap", 14, 30),
        ("fact", 130, 30),
    ]
    assert evaluate(c.win_condition, _ctx(own, tick=3000))
    # FAIL: only 2 pbox — bar is 3.
    own_short = [
        ("fact", 10, 30),
        ("pbox", 12, 30), ("pbox", 12, 31),
        ("weap", 14, 30),
        ("fact", 130, 30),
    ]
    assert not evaluate(c.win_condition, _ctx(own_short, tick=3000))
    # FAIL: 2nd fact outside east region.
    own_off = [
        ("fact", 10, 30),
        ("pbox", 12, 30), ("pbox", 12, 31), ("pbox", 12, 32),
        ("weap", 14, 30),
        ("fact", 100, 30),  # > radius 8 from (130,30)
    ]
    assert not evaluate(c.win_condition, _ctx(own_off, tick=3000))
    assert evaluate(c.fail_condition, _ctx(own, tick=7300))


def test_predicates_hard_requires_second_proc():
    c = compile_level(load_pack(PACK_PATH), "hard")
    base = [
        ("fact", 10, 22),
        ("pbox", 12, 22), ("pbox", 12, 23), ("pbox", 12, 24),
        ("weap", 14, 22),
        ("fact", 130, 30),
        ("proc", 6, 18),
    ]
    # FAIL: phase 4 missing — only 1 proc.
    assert not evaluate(c.win_condition, _ctx(base, tick=3000))
    # WIN: phase 4 satisfied — 2 procs (the original + a second).
    base2 = base + [("proc", 130, 32)]
    assert evaluate(c.win_condition, _ctx(base2, tick=3000))


# ── engine-driven scripted policies ─────────────────────────────────


def _find_mcv(rs):
    for u in rs.get("units_summary", []) or []:
        if str(u.get("type", "")).lower() == "mcv":
            return u
    return None


def _find_harv(rs):
    for u in rs.get("units_summary", []) or []:
        if str(u.get("type", "")).lower() == "harv":
            return u
    return None


def _base_xy(rs):
    """Return the base #1 (west-side) fact (cell_x, cell_y), preferring
    the smaller cell_x (the home base)."""
    facts = [
        b for b in (rs.get("own_buildings", []) or [])
        if str(b.get("type", "")).lower() == "fact"
    ]
    if not facts:
        return None
    return min(facts, key=lambda b: b["cell_x"])


def _stall_policy(rs, Command):
    """Idles every turn — must LOSE (clock + patrol attrition)."""
    return [Command.observe()]


def _pure_expand_skip_defence_policy():
    """Race the MCV east + deploy — never build pbox / weap. The
    chain cannot advance past phase 1 (pbox); the patrol also chews
    the base over time. Must LOSE."""

    def pol(rs, Command):
        mcv = _find_mcv(rs)
        if mcv is None:
            return [Command.observe()]
        tx, ty = 131, 31
        dx = mcv["cell_x"] - tx
        dy = mcv["cell_y"] - ty
        if dx * dx + dy * dy <= 36:
            return [Command.deploy([str(mcv["id"])])]
        return [Command.move_units([str(mcv["id"])], target_x=tx, target_y=ty)]

    return pol


def _pure_defence_no_expand_policy(pbox_target: int):
    """Build pbox + weap forever — never move the MCV. The chain
    stalls at phase 3 (no 2nd fact in east region); clock expires
    ⇒ LOSS."""
    state = {"placed_pbox": 0, "weap_attempts": 0, "harv_kicked": False}

    def pol(rs, Command):
        ob = rs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = rs.get("production", []) or []
        cash = int(rs.get("cash", 0) or 0)
        base = _base_xy(rs)
        bx = base["cell_x"] if base else 10
        by = base["cell_y"] if base else 30
        cmds = []
        # Kick the harv at the local mine (west of base).
        if not state["harv_kicked"]:
            harv = _find_harv(rs)
            if harv:
                cmds.append(Command.harvest([str(harv["id"])], bx - 8, by))
                state["harv_kicked"] = True
        pbox_have = sum(1 for b in ob if b["type"] == "pbox")
        if pbox_have < pbox_target and "pbox" not in prod and cash >= 600:
            cmds.append(Command.build("pbox"))
        if pbox_have < pbox_target:
            i = state["placed_pbox"]
            row = -3 + 2 * (i % 4)
            col = 3 + (i // 4)
            cmds.append(Command.place_building(
                "pbox", bx + col, by + row
            ))
            state["placed_pbox"] += 1
        if (
            pbox_have >= pbox_target
            and "weap" not in own_b
            and "weap" not in prod
            and cash >= 2000
        ):
            cmds.append(Command.build("weap"))
        if pbox_have >= pbox_target and "weap" not in own_b:
            i = state["weap_attempts"]
            cmds.append(Command.place_building(
                "weap", bx + 8 + (i % 6), by - 4 + (i // 6)
            ))
            state["weap_attempts"] += 1
        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return pol


def _intended_secure_expand_with_tech_policy(pbox_target: int, hard: bool = False):
    """The intended REASONING capability play:
       turn 1: kick the harvester onto the local mine
       phase 1: queue pbox one at a time until ≥pbox_target standing
                AT THE SAME TIME, walk the MCV partway east while
                income builds
       phase 2: once phase 1 done, build weap
       phase 3: once phase 2 done, move MCV the rest of the way and
                deploy at (131,31) → fact lands at (130,30) inside
                the east target region
       phase 4 (hard only): build a SECOND proc at the new east base
                (place it near the new east fact)
    """
    state = {
        "placed_pbox": 0,
        "weap_attempts": 0,
        "proc2_attempts": 0,
        "harv_kicked": False,
        "mcv_deployed": False,
    }

    def pol(rs, Command):
        ob = rs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = rs.get("production", []) or []
        cash = int(rs.get("cash", 0) or 0)
        base = _base_xy(rs)
        bx = base["cell_x"] if base else 10
        by = base["cell_y"] if base else 30
        mcv = _find_mcv(rs)
        cmds = []

        # Kick the harv at the local mine (west of base).
        if not state["harv_kicked"]:
            harv = _find_harv(rs)
            if harv:
                cmds.append(Command.harvest([str(harv["id"])], bx - 8, by))
                state["harv_kicked"] = True

        # ── Phase 1: pbox first ──
        pbox_have = sum(1 for b in ob if b["type"] == "pbox")
        in_q_pbox = "pbox" in prod
        if pbox_have < pbox_target and not in_q_pbox and cash >= 600:
            cmds.append(Command.build("pbox"))
        if pbox_have < pbox_target:
            i = state["placed_pbox"]
            row = -3 + 2 * (i % 4)
            col = 3 + (i // 4)
            cmds.append(Command.place_building(
                "pbox", bx + col, by + row
            ))
            state["placed_pbox"] += 1

        # ── Phase 2: weap (once phase-1 is satisfied) ──
        if (
            pbox_have >= pbox_target
            and "weap" not in own_b
            and "weap" not in prod
            and cash >= 2000
        ):
            cmds.append(Command.build("weap"))
        if pbox_have >= pbox_target and "weap" not in own_b:
            i = state["weap_attempts"]
            cmds.append(Command.place_building(
                "weap", bx + 8 + (i % 6), by - 4 + (i // 6)
            ))
            state["weap_attempts"] += 1

        # ── Phase 3: drive the MCV east + deploy at (131,31). Start
        # moving as soon as Phase 1 is observed-locked (the then-
        # composite is greedy so we can already be in motion). The
        # MCV deploy itself is gated until weap is up — otherwise a
        # too-early deploy would attempt to latch clause 3 before
        # clause 2 and the chain would not advance.
        if mcv is not None and not state["mcv_deployed"]:
            tx, ty = 131, 31
            dx = mcv["cell_x"] - tx
            dy = mcv["cell_y"] - ty
            dist2 = dx * dx + dy * dy
            in_range = dist2 <= 36
            # Move toward the target as soon as the chain is being
            # worked; only deploy once weap is up (phase 2 latched).
            if in_range and "weap" in own_b:
                cmds.append(Command.deploy([str(mcv["id"])]))
                state["mcv_deployed"] = True
            elif not in_range:
                cmds.append(Command.move_units(
                    [str(mcv["id"])], target_x=tx, target_y=ty
                ))

        # ── Phase 4 (hard only): build a SECOND proc near the new
        # east base. Only attempt once the east fact exists (phase 3
        # latched).
        if hard:
            proc_have = sum(1 for b in ob if b["type"] == "proc")
            east_fact = any(
                b["type"] == "fact" and b["cell_x"] >= 100 for b in ob
            )
            if (
                east_fact
                and proc_have < 2
                and "proc" not in prod
                and cash >= 1400
            ):
                cmds.append(Command.build("proc"))
            if east_fact and proc_have < 2:
                i = state["proc2_attempts"]
                # Place near the new east fact (which lives near
                # (130,30) after deploy).
                cmds.append(Command.place_building(
                    "proc", 132 + (i % 4), 32 + (i // 4)
                ))
                state["proc2_attempts"] += 1

        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return pol


# ── Engine-bound tests (parameterised over seeds 1..4) ───────────────


def _pbox_target(level: str) -> int:
    return {"easy": 2, "medium": 3, "hard": 3}[level]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_secure_expand_with_tech_wins(level, seed):
    """The intended secure-expand-with-tech policy must WIN on every
    (level, seed). Load-bearing solvency test."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(
        c,
        _intended_secure_expand_with_tech_policy(
            _pbox_target(level), hard=(level == "hard")
        ),
        seed=seed,
    )
    own_b = res.signals.own_building_types
    assert res.outcome == "win", (
        f"intended secure-expand-with-tech must WIN on {level} s={seed}; "
        f"got {res.outcome} turns={res.turns} "
        f"own_buildings={own_b} cash={res.signals.cash} "
        f"units_lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSE on every (level, seed)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall_policy, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_expand_skip_defence_loses(level, seed):
    """Race-MCV-east-and-deploy without ever building pbox/weap must
    LOSE: the then-chain cannot advance past phase 1 (pbox), and the
    patrol bot is dripping pressure on the undefended base."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _pure_expand_skip_defence_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"pure-expand-skip-defence must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_defence_no_expand_loses(level, seed):
    """Build pbox + weap forever — never move the MCV east. The
    then-chain stalls at phase 3 (no 2nd fact in east region); clock
    expires ⇒ LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(
        c, _pure_defence_no_expand_policy(_pbox_target(level)), seed=seed
    )
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"pure-defence-no-expand must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin per seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, _stall_policy, seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
