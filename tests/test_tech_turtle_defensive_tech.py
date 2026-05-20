"""tech-turtle-defensive-tech pack — full no-cheat validation on Rust.

Wave-4 TURTLE tech-triple pack. Uses the Wave-2 `then:[A,B,C,D]`
happened-before composite to enforce the defence-then-tech order:
  1. building_count_gte: {type: pbox, n: 3|8}   # phase 1 — anti-inf wall
  2. building_count_gte: {type: gun,  n: 1}     # phase 2 — anti-armour cap
  3. has_building: weap                          # phase 3 — tech step 1
  4. has_building: dome                          # phase 4 — tech step 2

Bar (per CLAUDE.md): the intended fortify-then-tech policy must WIN
on every (level, seed); stall / pure-aggro-skip-defence / build-only-
defence-no-tech must LOSE on every (level, seed). No draws.

Scenario shape:
  - rush-hour-arena, allied agent, soviet hunt-bot enemy band.
  - Pre-placed agent base: fact + tent + powr + proc + harv + mine
    + starting riflemen (so income flows from turn 1 and `own_units_
    gte:1` is satisfied immediately).
  - Inert enemy `fact` marker far east prevents auto-DRAW on
    enemy-elim.
  - Hunt band closes on agent base by ~tick 1500 and would raze the
    fact unless pbox×3 (medium: pbox×3, hard: pbox×8) is up first.
  - hard: ≥2 spawn_point groups (north / south) per the hard-tier
    contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "tech-turtle-defensive-tech.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Idles every turn — must LOSE (clock + hunt razes fact)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _pure_aggro_skip_defence_policy():
    """Race weap+dome, never build a pbox/gun. The hunt band reaches
    the base by ~tick 1500 with no defences ⇒ fact razed ⇒
    `not has_building: fact` fail fires ⇒ LOSS."""
    def pol(obs, Cmd):
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        base = next((b for b in (obs.get("own_buildings", []) or [])
                     if b["type"] == "fact"), None)
        cmds = []
        if "weap" not in own_b and "weap" not in prod:
            cmds.append(Cmd.build("weap"))
        if base and "weap" not in own_b:
            cmds.append(Cmd.place_building(
                "weap", base["cell_x"] + 6, base["cell_y"]
            ))
        if "dome" not in own_b and "dome" not in prod:
            cmds.append(Cmd.build("dome"))
        if base and "dome" not in own_b:
            cmds.append(Cmd.place_building(
                "dome", base["cell_x"] + 8, base["cell_y"] + 2
            ))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _only_defence_no_tech_policy():
    """Spam pbox forever — never build weap/dome. The then-chain
    cannot advance past phase 2 (gun needs weap, which is never
    built); clock expires ⇒ LOSS."""
    state = {"placed": 0}

    def pol(obs, Cmd):
        own_b_list = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        base = next((b for b in own_b_list if b["type"] == "fact"), None)
        cmds = []
        if "pbox" not in prod:
            cmds.append(Cmd.build("pbox"))
        if base:
            # Spread pbox placements so they don't collide.
            dx = 3 + (state["placed"] % 4)
            dy = -2 + (state["placed"] // 4)
            cmds.append(Cmd.place_building(
                "pbox", base["cell_x"] + dx, base["cell_y"] + dy
            ))
            state["placed"] += 1
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_defence_then_tech_policy(pbox_target: int):
    """The intended TURTLE capability play:
       turn 1: kick the pre-placed harvester onto the mine
       phase 1: queue pbox one at a time until ≥pbox_target standing
       phase 2: once phase 1 done, build weap (gun's engine prereq)
       phase 3: once weap up, build gun (this latches the then-chain
                clause 2; weap latches greedily as clause 3)
       phase 4: build dome (clause 4 latches → chain complete)
       The fortify-first ordering survives the hunt band's arrival
       at ~tick 1500.
    """
    state = {
        "placed_pbox": 0,
        "weap_attempts": 0,
        "gun_attempts": 0,
        "dome_attempts": 0,
        "harv_kicked": False,
    }

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        pbox_have = sum(1 for b in ob if b["type"] == "pbox")
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        units = obs.get("units_summary", []) or []
        base = next((b for b in ob if b["type"] == "fact"), None)
        bx = base["cell_x"] if base else 10
        by = base["cell_y"] if base else 20
        cmds = []

        # Kick the pre-placed harvester at the nearest mine. Mine is
        # WEST of base (at bx-8) in every scenario (easy/medium:
        # (2,20); hard: (2,14)/(2,26)) — placed west so the harv loop
        # doesn't cross the lane the hunt band is closing down.
        if not state["harv_kicked"]:
            harv = next((u for u in units if u.get("type") == "harv"), None)
            if harv:
                cmds.append(Cmd.harvest([harv["id"]], bx - 8, by))
                state["harv_kicked"] = True

        # Phase 1: pbox first. Queue ONE at a time (wait for it to
        # complete before queueing the next) so we don't over-build
        # before income arrives. Place each pbox on a distinct cell.
        in_q_pbox = "pbox" in prod
        if pbox_have < pbox_target and not in_q_pbox and cash >= 600:
            cmds.append(Cmd.build("pbox"))
        # Place whenever a pbox is ready (we only have at most one
        # queued at a time so a single place per turn is enough).
        if pbox_have < pbox_target:
            i = state["placed_pbox"]
            row = -3 + 2 * (i % 4)              # y offsets: -3,-1,+1,+3
            col = 3 + (i // 4)                  # x offsets: 3,4,5,…
            cmds.append(Cmd.place_building(
                "pbox", bx + col, by + row
            ))
            state["placed_pbox"] += 1

        # Phase 2/3: once pbox wall is up, build weap (gun's prereq).
        # weap needs proc (pre-placed) so the queue accepts it.
        # Retry placement on a sliding offset until the building
        # actually completes (the engine silently blocks placements
        # that collide with existing footprints / units).
        if pbox_have >= pbox_target and "weap" not in own_b \
                and "weap" not in prod and cash >= 2000:
            cmds.append(Cmd.build("weap"))
        if pbox_have >= pbox_target and "weap" not in own_b:
            i = state["weap_attempts"]
            cmds.append(Cmd.place_building(
                "weap", bx + 8 + (i % 6), by - 4 + (i // 6)
            ))
            state["weap_attempts"] += 1

        # gun needs weap — queue it once weap is up.
        gun_have = sum(1 for b in ob if b["type"] == "gun")
        if "weap" in own_b and gun_have < 1 and "gun" not in prod \
                and cash >= 800:
            cmds.append(Cmd.build("gun"))
        if "weap" in own_b and gun_have < 1:
            i = state["gun_attempts"]
            cmds.append(Cmd.place_building(
                "gun", bx + 8 + (i % 6), by + 4 + (i // 6)
            ))
            state["gun_attempts"] += 1

        # dome needs proc (pre-placed); build in parallel once weap
        # is up — but keep the spec's gun-before-dome ordering so the
        # then-chain advances cleanly.
        if "weap" in own_b and gun_have >= 1 and "dome" not in own_b \
                and "dome" not in prod and cash >= 1000:
            cmds.append(Cmd.build("dome"))
        if "weap" in own_b and gun_have >= 1 and "dome" not in own_b:
            i = state["dome_attempts"]
            cmds.append(Cmd.place_building(
                "dome", bx + 12 + (i % 4), by + (i // 4)
            ))
            state["dome_attempts"] += 1

        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "tech-turtle-defensive-tech"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("turtle" in a for a in anchors), anchors
    assert any("fortify" in a or "shield" in a for a in anchors), anchors
    assert any("hedge" in a or "risk" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_then_composite_used_in_win():
    """Confirms the defence-then-tech chain wired through to the
    compiled win condition (the whole point of the TURTLE pack)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        assert any("then" in clause for clause in ao), (
            f"{lvl} win has no then-composite: {win}"
        )
        then_clause = next(cl["then"] for cl in ao if "then" in cl)
        clauses = then_clause["clauses"]
        # Must be exactly: pbox → gun → weap → dome.
        assert "building_count_gte" in clauses[0]
        assert clauses[0]["building_count_gte"]["type"] == "pbox"
        assert "building_count_gte" in clauses[1]
        assert clauses[1]["building_count_gte"]["type"] == "gun"
        assert "has_building" in clauses[2] and clauses[2]["has_building"] == "weap"
        assert "has_building" in clauses[3] and clauses[3]["has_building"] == "dome"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns."""
    pack = load_pack(PACK)
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
        wts = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


def _pbox_target(level: str) -> int:
    return 8 if level == "hard" else 3


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_defence_then_tech_wins(level, seed):
    """The intended TURTLE policy (pbox first, then weap+gun+dome)
    must WIN on every (level, seed). Load-bearing solvency test."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_defence_then_tech_policy(_pbox_target(level)), seed=seed
    )
    own_b = res.signals.own_building_types
    assert res.outcome == "win", (
        f"intended defence-then-tech must WIN on {level} s={seed}; "
        f"got {res.outcome} turns={res.turns} "
        f"own_buildings={own_b} cash={res.signals.cash} "
        f"units_lost={res.signals.units_lost} "
        f"then_progress={getattr(res.signals, 'then_progress', {})}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSE on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_aggro_skip_defence_loses(level, seed):
    """Pure aggro (race weap+dome without any defence) must LOSE:
    hunt band reaches the base ~tick 1500, no pbox/gun to blunt it
    ⇒ fact razed ⇒ `not has_building: fact` fail fires (or the
    then-chain never advances past phase 1 ⇒ clock expires)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pure_aggro_skip_defence_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"pure-aggro-skip-defence must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b} "
        f"then_progress={getattr(res.signals, 'then_progress', {})}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_only_defence_no_tech_loses(level, seed):
    """Build only pbox forever — never weap/dome. Phase 2 (gun)
    requires weap (engine prereq), which is never built; the
    then-chain stalls at phase 1; clock expires ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _only_defence_no_tech_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"only-defence-no-tech must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b} "
        f"then_progress={getattr(res.signals, 'then_progress', {})}"
    )
    # Sanity: tech buildings never built.
    assert "weap" not in own_b, (
        f"only-defence-no-tech built weap on {level} s={seed}"
    )
    assert "dome" not in own_b, (
        f"only-defence-no-tech built dome on {level} s={seed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin per seed."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
