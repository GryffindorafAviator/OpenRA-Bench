"""No-cheat + solvency proof for `econ-recover-from-zero-cash` (Group F
turtle-recovery / bankruptcy turnaround from a minimum-viable kit and
zero working capital).

The pack frames a RECOVERY-FROM-SETBACK decision: the agent inherits a
minimum viable kit (fact + proc + one harv + one mine) and **$0 cash**.
It must commit to harvesting first (the only income channel), then
reinvest the accumulated cash on the next productive asset (a war
factory + a second/third harvester) so the income channel actually
scales past a non-trivial revenue bar.

For every level + every hard seed (1-4):
  * the INTENDED chain (harvest → save 2000 → build weap → build harv →
    move+harvest new harv [→ build 3rd harv on hard]) WINS;
  * STALL (only `observe`) LOSES every tier — no harvest, EV stays at 0,
    clock bites as a reachable timeout LOSS;
  * HARVEST-ONLY (run the starting harv but never build anything) is
    the EASY floor (wins easy) but LOSES medium/hard on the 2-harv /
    3-harv structural clause regardless of EV;
  * BUILD-ARMY-FROM-ZERO (harvest → powr → tent → spam e1 instead of
    weap → harv) LOSES every tier — no weap means no harv can be built,
    and `tent`+`e1` spend drains the cash that should have gone to the
    income-scaling chain;
  * BUILD-TOO-SOON (harvest → spend cash on a SECOND refinery instead of
    weap+harv) LOSES medium/hard — cash drained on the wrong asset
    starves the 2-harv requirement.

The 4 lazy plays + 1 intended × 3 levels × 4 seeds gives the full
no-defect / no-cheat coverage demanded by CLAUDE.md.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-recover-from-zero-cash.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── helpers ────────────────────────────────────────


def _patches_for_fact(fy: int) -> list[tuple[int, int]]:
    """easy/medium have one patch at (22,18); hard has spawn-local
    near patches: NORTH spawn → (22,14)+(22,16); SOUTH → (22,26)+(22,28)."""
    if abs(fy - 18) <= 2:
        return [(22, 18)]
    if fy < 18:
        return [(22, 14), (22, 16)]
    return [(22, 26), (22, 28)]


def _own_fact(bldgs):
    """Return the AGENT-owned fact (skip the far enemy marker)."""
    for b in bldgs:
        if b.get("type") == "fact" and b.get("owner", "agent") != "enemy":
            cx = b.get("cell_x", 0)
            # The enemy marker is at (120,20) — filter it by x.
            if cx < 100:
                return b
    return None


def _ev(res):
    return res.signals.cash + res.signals.resources


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Idle: never harvests, never builds. EV stays at 0 → bar unmet →
    reachable timeout LOSS."""
    return [Command.observe()]


def _harvest_only_factory():
    """Run the starting harv and queue a minimum-cost power plant (the
    easy-tier building bar). WINS easy (one harv income + powr clears
    the 1500 EV bar AND the building_count_gte:powr clause) but LOSES
    medium/hard (the 2-harv / 3-harv structural clause fails regardless
    of EV). The pre-placed harv auto-harvests once it has line-of-sight
    to the patch; we issue a single explicit `harvest` order to nudge
    the auto-cycle (re-issuing move/harvest every turn cancels the
    auto-cycle and stalls income — engine quirk noted in pack header)."""

    state = {"sent": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        fact_b = _own_fact(bldgs)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        patches = _patches_for_fact(fy)
        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        cmds = []
        for i, (uid, _cx, _cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["sent"]:
                cmds.append(Command.harvest([uid_s], px, py))
                state["sent"].add(uid_s)
        # Build a single cheap powr to clear the easy-tier building bar.
        if "powr" not in own_types:
            if cash >= 300 and "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 2, fy - 3))
        return cmds if cmds else [Command.observe()]

    return policy


def _build_army_factory():
    """Harvest a bit (one-shot order — repeated orders cancel the
    auto-cycle, see engine quirk in pack header), then powr → tent →
    spam e1 instead of weap+harv. Without weap, no harv can be built →
    2-harv clause fails → LOSS on every tier (even easy, where the
    army spend drains the EV bar and the building bar requires powr
    AND nothing else gates the army track — the policy still queues
    powr because tent's prereq IS powr, so the easy building-bar
    failure has to come from the EV bar instead: army spending pulls
    cash below the EV floor before the 70-turn clock runs out)."""

    state = {"sent": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        fact_b = _own_fact(bldgs)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        patches = _patches_for_fact(fy)
        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        cmds = []
        for i, (uid, _cx, _cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["sent"]:
                cmds.append(Command.harvest([uid_s], px, py))
                state["sent"].add(uid_s)
        if "powr" not in own_types:
            if cash >= 300 and "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 2, fy - 3))
        elif "tent" not in own_types:
            if cash >= 400 and "tent" not in prod:
                cmds.append(Command.build("tent"))
            cmds.append(Command.place_building("tent", fx + 2, fy + 3))
        elif cash >= 100 and "e1" not in prod:
            cmds.append(Command.build("e1"))
        return cmds if cmds else [Command.observe()]

    return policy


def _build_too_soon_factory():
    """Harvest (one-shot), then spend the FIRST accumulated cash on a
    SECOND refinery (the wrong scaling asset) instead of weap+harv.
    The income channel does NOT scale (single harv still) → the 2-harv
    clause fails on medium/hard → LOSS."""

    state = {"sent": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        fact_b = _own_fact(bldgs)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        patches = _patches_for_fact(fy)
        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        cmds = []
        for i, (uid, _cx, _cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["sent"]:
                cmds.append(Command.harvest([uid_s], px, py))
                state["sent"].add(uid_s)
        # Always try to queue a second proc and place it — the engine
        # may block on prereqs but that's the point (spending intent on
        # the wrong asset, not weap/harv).
        n_proc = sum(1 for b in bldgs if b.get("type") == "proc")
        if n_proc < 2:
            if cash >= 1400 and "proc" not in prod:
                cmds.append(Command.build("proc"))
            cmds.append(Command.place_building("proc", fx + 5, fy + 1))
        return cmds if cmds else [Command.observe()]

    return policy


def _intended_factory():
    """Intended recovery chain: harvest with starting harv → save 2000 →
    build('weap') + place → save 1400 → build('harv') → harvest the new
    harv (one-shot). On hard, build a 3rd harv after the 2nd. Also
    builds a single powr to satisfy the easy-tier building bar (no-op
    cost on medium/hard since powr isn't a win clause there). WINS
    every tier × every seed."""

    state = {"sent": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        fact_b = _own_fact(bldgs)
        proc_b = next((b for b in bldgs if b.get("type") == "proc"), None)
        if fact_b is None or proc_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        patches = _patches_for_fact(fy)
        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        cmds = []
        # One-shot harvest order per harv (repeated orders cancel the
        # auto-cycle — engine quirk; see pack header note 5).
        for i, (uid, _cx, _cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["sent"]:
                cmds.append(Command.harvest([uid_s], px, py))
                state["sent"].add(uid_s)
        # Build a cheap powr to clear the easy-tier building bar (no
        # win-clause cost on medium/hard; the spend is harmless).
        if "powr" not in own_types:
            if cash >= 300 and "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 2, fy - 3))
        # Then queue weap as soon as cash allows; spam place_building
        # each turn (engine ignores PLACE until production completes).
        if "weap" not in own_types:
            if cash >= 2000 and "weap" not in prod:
                cmds.append(Command.build("weap"))
            cmds.append(Command.place_building("weap", fx + 2, fy + 3))
        else:
            # Determine how many harvs we want: 2 for single-patch
            # tiers (easy/medium), 3 for the two-patch hard tier.
            want_harv = 2 if len(patches) == 1 else 3
            if len(harvs) < want_harv and cash >= 1400 and "harv" not in prod:
                cmds.append(Command.build("harv"))
        return cmds if cmds else [Command.observe()]

    return policy


def _run(level, policy_or_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "bespoke 48x40 arena must compile"
    pol = (
        policy_or_factory()
        if callable(policy_or_factory)
        and policy_or_factory.__name__.endswith("factory")
        else policy_or_factory
    )
    return c, run_level(c, pol, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_required_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-recover-from-zero-cash"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # Spec-required tools (Wave-5 spec).
    for t in ("observe", "build", "place_building", "harvest", "move_units", "stop"):
        assert t in tools, f"missing tool {t} in {tools}"


def test_starting_cash_is_zero_on_every_level():
    """The pack headline is recovery FROM zero cash — every tier must
    start with 0 cash so the income-first → reinvest decision is real."""
    pack = load_pack(PACK)
    for L in LEVELS:
        c = compile_level(pack, L)
        assert c.starting_cash == 0, (
            f"{L}: starting_cash must be 0 (got {c.starting_cash})"
        )


def test_benchmark_anchor_lists_turtle_and_turnaround():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "sc2" in blob, anchors
    assert "turtle" in blob or "recovery" in blob, anchors
    assert "bankruptcy" in blob or "turnaround" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: after_ticks in
    fail_condition must be reachable within max_turns (tick ≈ 93 +
    90·(max_turns − 1)). within_ticks + 1 == after_ticks idiom."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    after_ticks = int(c.fail_condition.model_dump()["any_of"][0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    within_clauses = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in within_clauses if "within_ticks" in x)
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt + 1})"
    )


def test_hard_has_two_spawn_groups_for_base():
    """Hard tier contract: ≥2 distinct agent spawn_point groups
    (NORTH / SOUTH base) so seed round-robin varies the start cell and
    a memorised opening can't generalise."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WIN bar ───────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_chain_wins(level, seed):
    """Intended recovery (harvest → weap → harv [+ harv on hard]) WINS
    every tier × every seed."""
    c, r = _run(level, _intended_factory, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended chain should WIN, got {r.outcome}; "
        f"ev={_ev(r)}, cash={r.signals.cash}, harvs={r.signals.harvesters}, "
        f"types={r.signals.own_building_types}, tick={r.signals.game_tick}"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall (only `observe`) must LOSE every tier × every seed. Note
    the engine auto-harvests pre-placed harvs that already sit near a
    pre-placed proc — so a stall play DOES accrue passive cash (the
    auto-cycle keeps running). The win predicates have been structured
    so stall still LOSES on each tier:
      * easy: building_count_gte:powr (stall never builds powr → LOSS)
      * medium/hard: unit_type_count_gte:harv 2/3 (stall never builds
        a 2nd/3rd harv → LOSS)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_harvest_only_is_easy_floor(seed):
    """Harvest-only with the starting harv is the EASY FLOOR — it
    clears the loose 1500 bar without needing a 2nd harv."""
    c, r = _run("easy", _harvest_only_factory, seed=seed)
    assert r.outcome == "win", (
        f"easy seed{seed} harvest-only should be the FLOOR and WIN; "
        f"got {r.outcome} (ev={_ev(r)})"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_harvest_only_loses_medium_and_hard(level, seed):
    """Harvest-only LOSES medium/hard — the 2-harv / 3-harv structural
    clause fails regardless of EV. This is the discriminator that the
    recovery chain (build weap → build harv) is actually exercised."""
    c, r = _run(level, _harvest_only_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} harvest-only must LOSE on the 2/3-harv "
        f"clause; got {r.outcome} (ev={_ev(r)}, "
        f"harvs={r.signals.harvesters}, types={r.signals.own_building_types})"
    )
    # Harvester count never grew past the pre-placed one.
    assert r.signals.harvesters <= 1, (
        f"{level}: harvest-only should not produce more harvs; "
        f"harvs={r.signals.harvesters}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_army_loses(level, seed):
    """Harvest → powr → tent → spam e1 (NEVER builds weap → no harv)
    must LOSE every tier — easy on the EV bar (army drain), medium/hard
    on the 2/3-harv clause AND the EV bar."""
    c, r = _run(level, _build_army_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} build-army must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, types={r.signals.own_building_types})"
    )
    # Wrong-path commitment confirmed: tent landed but weap never did.
    assert "tent" in r.signals.own_building_types, (
        f"{level}: build-army should commit to tent; "
        f"types={r.signals.own_building_types}"
    )
    assert "weap" not in r.signals.own_building_types, (
        f"{level}: build-army should NEVER build weap; "
        f"types={r.signals.own_building_types}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_build_too_soon_loses_medium_and_hard(level, seed):
    """Build-too-soon = spend cash on a 2nd refinery (the wrong scaling
    asset) instead of weap+harv. The income channel does NOT scale →
    the 2/3-harv clause fails on medium/hard → LOSS."""
    c, r = _run(level, _build_too_soon_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} build-too-soon must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, harvs={r.signals.harvesters}, "
        f"types={r.signals.own_building_types})"
    )
    # No weap (so no harv could be built) — the discriminator.
    assert "weap" not in r.signals.own_building_types, (
        f"{level}: build-too-soon must not happen to also build weap; "
        f"types={r.signals.own_building_types}"
    )
    assert r.signals.harvesters <= 1, (
        f"{level}: build-too-soon should not produce more harvs; "
        f"harvs={r.signals.harvesters}"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared
    spawn_point groups (NORTH y=14 / SOUTH y=26) so a memorised opening
    can't generalise."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended_factory(), seed=2)
    b = run_level(c, _intended_factory(), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b)), (
        f"determinism: {(a.outcome, a.turns, _ev(a))} vs "
        f"{(b.outcome, b.turns, _ev(b))}"
    )
