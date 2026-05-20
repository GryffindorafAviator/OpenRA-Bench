"""mfb-tech-base-vs-economy-base pack — full no-cheat validation.

REASONING pack — role-specialized 2-base layout. The agent owns a
fact + powr + mcv at NW (20,20) and SE (140,60); a preplaced harv
parks at one of them (NW by default; flips per spawn on hard). The
big ore cluster sits next to SE. The win predicate is a cross-
region split — weap-in-NW-region AND proc-in-SE-region AND ≥2 harvs
AND economy ≥ 2500, all within the clock — so bunching both
production lines at one pole or stalling never satisfies it.

Bar (per CLAUDE.md):
  • intended split (weap @ NW + proc @ SE) WINS every (level, seed).
  • stall LOSES every (level, seed).
  • tech-and-econ bunched at NW (both production lines placed inside
    the NW region) LOSES every (level, seed).
  • tech-and-econ bunched at SE (both placed inside the SE region)
    LOSES every (level, seed).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mfb-tech-base-vs-economy-base.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Anchor coordinates — must stay in sync with the YAML.
NW = (20, 20)
SE = (140, 60)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _split_policy():
    """The intended capability play.

    1. Build proc; place at SE near (137, 58) — satisfies
       SE-econ-clause AND spawns the free Ore Truck on the big ore
       patch (drives unit_type_count_gte:harv,n:2).
    2. Once proc is owned, build weap; place at NW near (23, 22) —
       satisfies NW-tech-clause.
    3. Any preplaced harv gets a `harvest` order at the near patch
       (NW: 35,20; SE: 125,60) so the economy_value bar fills from
       both sites in parallel.

    `place_building` is RETRIED every turn while the structure is
    in the queue but not yet owned — the engine emits "PLACE
    BLOCKED: not completed in queue" on early attempts and lands it
    once the build clock finishes; we keep firing until the building
    surfaces in `own_buildings`. Same idiom as economy/mcv-deploy
    scripted policies elsewhere in the suite.
    """
    state = {"harv_dispatched": set()}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        cmds = []
        # Step 1: queue proc; keep firing place_building at SE until
        # it lands in own_buildings.
        if "proc" not in own_b and "proc" not in prod:
            cmds.append(Cmd.build("proc"))
        if "proc" not in own_b and "proc" in prod:
            cmds.append(Cmd.place_building("proc", 137, 58))
        # Step 2: once proc is owned, queue weap; same retry idiom
        # to keep placing at NW until it surfaces.
        if "proc" in own_b and "weap" not in own_b and "weap" not in prod:
            cmds.append(Cmd.build("weap"))
        if "proc" in own_b and "weap" not in own_b and "weap" in prod:
            cmds.append(Cmd.place_building("weap", 23, 22))
        # Step 3: dispatch every newly-visible harv at its nearer patch.
        units = obs.get("units_summary", []) or []
        for u in units:
            if u.get("type") != "harv":
                continue
            uid = str(u["id"])
            if uid in state["harv_dispatched"]:
                continue
            ux, uy = u.get("cell_x", 0), u.get("cell_y", 0)
            tgt = (35, 20) if abs(ux - 20) + abs(uy - 20) < abs(ux - 140) + abs(uy - 60) else (125, 60)
            cmds.append(Cmd.harvest([uid], tgt[0], tgt[1]))
            state["harv_dispatched"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _bunched_policy(anchor: tuple[int, int]):
    """Place BOTH proc AND weap at the SAME pole (NW or SE). This
    satisfies one region clause but never the other — clock expires
    → LOSS. The harv-dispatch logic still runs so the economy clause
    isn't the discriminator; the cross-region predicate is."""
    ax, ay = anchor
    state = {"harv_dispatched": set()}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        cmds = []
        if "proc" not in own_b and "proc" not in prod:
            cmds.append(Cmd.build("proc"))
        if "proc" not in own_b and "proc" in prod:
            cmds.append(Cmd.place_building("proc", ax - 3, ay - 2))
        if "proc" in own_b and "weap" not in own_b and "weap" not in prod:
            cmds.append(Cmd.build("weap"))
        if "proc" in own_b and "weap" not in own_b and "weap" in prod:
            cmds.append(Cmd.place_building("weap", ax + 3, ay + 2))
        units = obs.get("units_summary", []) or []
        nearest_ore = (125, 60) if anchor == SE else (35, 20)
        for u in units:
            if u.get("type") != "harv":
                continue
            uid = str(u["id"])
            if uid in state["harv_dispatched"]:
                continue
            cmds.append(Cmd.harvest([uid], nearest_ore[0], nearest_ore[1]))
            state["harv_dispatched"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; no engine) ──────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "mfb-tech-base-vs-economy-base"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("industrial specialization" in a for a in anchors), anchors
    assert any("SC2" in a for a in anchors), anchors
    assert any("multi-warehouse" in a for a in anchors), anchors
    assert any("read-replica" in a or "write-primary" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start layout (binding contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_cross_region_predicate_in_win():
    """The whole point of the pack: WIN must require weap inside the
    NW region AND proc inside the SE region (the role-specialization
    teeth)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        clauses = win.get("all_of") or []
        # Look for both building_in_region clauses.
        weap_nw = any(
            ("building_in_region" in cl
             and cl["building_in_region"].get("type") == "weap"
             and cl["building_in_region"].get("x") == 20
             and cl["building_in_region"].get("y") == 20)
            for cl in clauses
        )
        proc_se = any(
            ("building_in_region" in cl
             and cl["building_in_region"].get("type") == "proc"
             and cl["building_in_region"].get("x") == 140
             and cl["building_in_region"].get("y") == 60)
            for cl in clauses
        )
        assert weap_nw, f"{lvl}: no weap@NW(20,20) building_in_region clause"
        assert proc_se, f"{lvl}: no proc@SE(140,60) building_in_region clause"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks reachable inside max_turns (engine advances ~90
    ticks/turn ⇒ reachable max = 93 + 90·(N-1))."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        wts: list[int] = []

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={level_def.max_turns}) — deadline never bites"
            )


# ── Engine-bound tests (seeds 1..4) ──────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_split_wins(level, seed):
    """The intended role-specialization policy (weap@NW + proc@SE
    + harvest both patches) MUST WIN on every (level, seed). This is
    the load-bearing test that the pack is solvable inside the budget
    by the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _split_policy(), seed=seed)
    assert res.outcome == "win", (
        f"intended split must WIN on {level} s={seed}; got {res.outcome} "
        f"(cash={res.signals.cash}, res={res.signals.resources}, "
        f"buildings={list(res.signals.own_building_types)})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSE on every (level, seed) — the fail_condition
    after_ticks clause bites at the turn budget; never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_bunched_at_nw_loses(level, seed):
    """Placing both proc AND weap inside the NW region satisfies the
    NW-weap clause but never the SE-proc clause → clock expires →
    LOSS on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _bunched_policy(NW), seed=seed)
    assert res.outcome == "loss", (
        f"bunched-NW must LOSE on {level} s={seed}; got {res.outcome} "
        f"(buildings={list(res.signals.own_building_types)})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_bunched_at_se_loses(level, seed):
    """Placing both proc AND weap inside the SE region satisfies the
    SE-proc clause but never the NW-weap clause → LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _bunched_policy(SE), seed=seed)
    assert res.outcome == "loss", (
        f"bunched-SE must LOSE on {level} s={seed}; got {res.outcome} "
        f"(buildings={list(res.signals.own_building_types)})"
    )
