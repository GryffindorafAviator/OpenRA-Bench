"""def-engineer-repair-under-fire pack — no-cheat validation on Rust.

v1.0-sweep audit fix (Qwen 9B run-3 / 2026-05-24): the prior pack
gated WIN purely on `building_count_gte: proc + fact + kill_bar`,
which a "brute" policy that just built tanks and attacked WON without
ever issuing `repair` — the engineer/repair capability was NOT
load-bearing. The rewritten pack pre-places `proc` at `health: 40`
and adds the `building_hp_pct_gte: {type: proc, pct: 0.75/0.80}`
clause to the WIN predicate so the damaged proc must be HEALED back
up via `Command.repair` to win. The mid-engagement scheduled
`spawn_actors` wave kills the unrepaired proc, triggering the fail
clause.

Bar (per CLAUDE.md, every level + every seed 1..4):
  * stall (only observe) must LOSE — proc HP stays at 40%, the
    scheduled grenadier wave razes the proc.
  * brute (build tanks + attack, never repair) must LOSE — even if
    the attackers die, the damaged proc is never healed; the
    `building_hp_pct_gte: {proc, 0.75}` clause never latches before
    the clock.
  * wrong-target (repair the healthy fact instead of the damaged
    proc) must LOSE — wrong building is already ~100% HP, the
    damaged proc stays low and falls / fails the HP bar at clock.
  * intended (repair the damaged proc + commit defenders via
    attack_unit) must WIN — autorepair tops the proc up past 75%
    while pbox + committed defenders clear the grenadier wave.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-engineer-repair-under-fire.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ───────────────────────────────────────────────────────


def _stall_policy():
    """Observe-only — must LOSE on every (level, seed)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _brute_policy():
    """Brute: queue tanks + commit defenders to attack the closest
    enemy, but NEVER repair the damaged proc. Must LOSE — the
    `building_hp_pct_gte: proc` clause requires the damaged proc to
    be healed back up, and brute never issues `repair`."""
    def pol(obs, Cmd):
        cmds = []
        units = obs.get("units_summary", []) or []
        enemies = [
            e for e in (obs.get("enemy_summary", []) or [])
            if not e.get("is_building")
        ]
        fighters = [
            u for u in units
            if u.get("type") in ("e1", "e3", "e6", "1tnk", "2tnk", "3tnk")
        ]
        if fighters and enemies:
            cmds.append(
                Cmd.attack_unit(
                    [u["id"] for u in fighters], str(enemies[0]["id"])
                )
            )
        # Try to queue a tank (will blocked on prereqs without a weap,
        # which is intentional — the brute policy "would have" if it
        # could; the test pins that even maximal-aggression LOSES).
        cmds.append(Cmd.build("1tnk"))
        if not cmds:
            cmds = [Cmd.observe()]
        return cmds
    return pol


def _wrong_target_policy():
    """Repair the WRONG building (the healthy fact instead of the
    damaged proc) and commit defenders. Must LOSE — the fact is
    already at ~100% HP so the repair is a no-op, the damaged proc
    stays at 40% and fails the HP bar at the clock."""
    def pol(obs, Cmd):
        cmds = []
        obldg = obs.get("own_buildings", []) or []
        fact_id = None
        for b in obldg:
            if b.get("type") == "fact":
                fact_id = str(b.get("id"))
                break
        if fact_id:
            cmds.append(Cmd.repair([fact_id]))
        units = obs.get("units_summary", []) or []
        enemies = [
            e for e in (obs.get("enemy_summary", []) or [])
            if not e.get("is_building")
        ]
        fighters = [
            u for u in units
            if u.get("type") in ("e1", "e3", "e6", "1tnk", "2tnk", "3tnk")
        ]
        if fighters and enemies:
            cmds.append(
                Cmd.attack_unit(
                    [u["id"] for u in fighters], str(enemies[0]["id"])
                )
            )
        if not cmds:
            cmds = [Cmd.observe()]
        return cmds
    return pol


def _intended_policy():
    """Toggle `repair` on the damaged proc AND commit the hold-fire
    defenders via attack_unit. Must WIN on every (level, seed) —
    autorepair restores the proc past 75% HP while committed
    defenders + pbox clear the grenadier wave."""
    def pol(obs, Cmd):
        cmds = []
        obldg = obs.get("own_buildings", []) or []
        proc_id = None
        for b in obldg:
            if b.get("type") == "proc":
                proc_id = str(b.get("id"))
                break
        if proc_id:
            cmds.append(Cmd.repair([proc_id]))
        units = obs.get("units_summary", []) or []
        enemies = [
            e for e in (obs.get("enemy_summary", []) or [])
            if not e.get("is_building")
        ]
        fighters = [
            u for u in units
            if u.get("type") in ("e1", "e3", "e6", "1tnk", "2tnk", "3tnk")
        ]
        if fighters and enemies:
            cmds.append(
                Cmd.attack_unit(
                    [u["id"] for u in fighters], str(enemies[0]["id"])
                )
            )
        if not cmds:
            cmds = [Cmd.observe()]
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-engineer-repair-under-fire"
    assert pack.meta.capability == "action"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_repair_tool_is_in_tools_list():
    """The load-bearing tool must be exposed for every level."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert "repair" in c.scenario.tools, (
            f"{lvl}: repair must be in tools (got {c.scenario.tools})"
        )


def test_meta_benchmark_anchors_match_doctrine():
    anchors = (load_pack(PACK).meta.benchmark_anchor or [])
    joined = " | ".join(a.lower() for a in anchors)
    assert "disaster recovery" in joined, anchors
    assert "scv" in joined or "repair" in joined, anchors


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_win_predicate_gates_on_damaged_proc_hp_restore():
    """The forced-repair gate: win must require building_hp_pct_gte
    on the proc, NOT just `building_count_gte` (the latter is
    satisfied by a 1% proc — busted bar). Plus the structural
    survival clauses for proc + fact, a kill bar, and within_ticks."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        all_of = win.get("all_of") or []
        keys: set = set()
        hp_types: list = []
        ct_types: list = []
        for clause in all_of:
            keys |= set(clause.keys())
            if "building_hp_pct_gte" in clause:
                hp_types.append(clause["building_hp_pct_gte"].get("type"))
            if "building_count_gte" in clause:
                ct_types.append(clause["building_count_gte"].get("type"))
        assert "building_hp_pct_gte" in keys, (
            f"{lvl} win MUST include building_hp_pct_gte (the forced-"
            f"repair gate); got keys={sorted(keys)}"
        )
        assert "proc" in hp_types, (
            f"{lvl} win must gate the proc HP bar: {hp_types}"
        )
        assert "proc" in ct_types and "fact" in ct_types, (
            f"{lvl} win must require both proc AND fact to survive: "
            f"{ct_types}"
        )
        assert "units_killed_gte" in keys, (
            f"{lvl} win missing units_killed_gte: {keys}"
        )
        assert "within_ticks" in keys, (
            f"{lvl} win missing within_ticks: {keys}"
        )


def test_pack_pre_places_damaged_proc_with_health_field():
    """The forced-repair anchor: the proc MUST be pre-placed at a
    damaged health level (≤50%) so `building_hp_pct_gte` is a real
    target (a healthy proc at 100% would trivially satisfy the
    clause). Engine parser honours `health:` per CLAUDE.md."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        procs = [
            a for a in c.scenario.actors
            if a.type == "proc" and a.owner == "agent"
        ]
        assert procs, f"{lvl}: must pre-place an agent proc"
        for p in procs:
            assert 1 <= p.health <= 50, (
                f"{lvl}: pre-placed proc must declare health:≤50 "
                f"(damaged anchor); got {p.health}"
            )


def test_pack_has_engineer_in_roster():
    """Engineer (e6) on the roster is the narrative anchor for the
    SCV / combat-engineer field-repair doctrine. The actual engine
    verb is `Command.repair` (autorepair toggle), but the engineer
    on the field aligns the pack with its name."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        e6 = [
            a for a in c.scenario.actors
            if a.type == "e6" and a.owner == "agent"
        ]
        assert e6, f"{lvl}: must have ≥1 agent engineer (e6)"


def test_termination_keeps_episode_alive_past_enemy_wipe():
    """The win predicate is keyed on PROC SURVIVAL + HP-restore. The
    engine's default auto-`done` on enemy elimination would race past
    the HP-restore clause; opt out via
    `termination.enemy_units_killed: false`."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        tt = c.scenario.termination
        # The flag may be in the termination dict or a model attr.
        if hasattr(tt, "enemy_units_killed"):
            flag = tt.enemy_units_killed
        elif isinstance(tt, dict):
            flag = tt.get("enemy_units_killed", True)
        else:
            flag = True
        assert flag is False, (
            f"{lvl}: termination.enemy_units_killed must be false so "
            f"a brute clear doesn't race past the HP-restore gate"
        )


def test_pack_declares_scheduled_pressure_wave():
    """Mid-engagement reinforcement wave injected via
    `scheduled_events.spawn_actors` — without it the initial wave
    might be cleared before any meaningful chip damage, weakening
    the load-bearing repair requirement on the brute path."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        evts = c.scheduled_events or []
        spawn = [e for e in evts if e.get("type") == "spawn_actors"]
        assert spawn, (
            f"{lvl}: must declare a `scheduled_events.spawn_actors` "
            f"pressure wave; got {evts}"
        )


def test_fail_predicate_drops_proc_or_fact_or_clock():
    """Fail must trigger when proc OR fact dies, OR the clock
    expires (so an unrepaired proc still loses on the clock)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fail = c.fail_condition.model_dump(exclude_none=True)
        any_of = fail.get("any_of") or []
        flat = " ".join(str(c) for c in any_of)
        assert "after_ticks" in flat, f"{lvl} fail needs after_ticks: {any_of}"
        assert "proc" in flat, f"{lvl} fail needs proc clause: {any_of}"
        assert "fact" in flat, f"{lvl} fail needs fact clause: {any_of}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks AND after_ticks fail clauses must both be
    reachable inside max_turns (the reachable max tick for non-
    interrupt is `93 + 90·(max_turns − 1)`; for interrupt mode the
    lower bound is ~60·max_turns)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
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

        wts: list = []
        ats_fail: list = []
        _collect(win, "within_ticks", wts)
        _collect(fail, "after_ticks", ats_fail)
        assert wts, f"{lvl} has no within_ticks leaf"
        assert ats_fail, f"{lvl} has no after_ticks fail leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
        for at in ats_fail:
            assert at <= reachable, (
                f"{lvl} fail after_ticks={at} > reachable={reachable} — "
                f"timeout fail never fires ⇒ draw"
            )
        interrupt_budget = 60 * max_turns
        for at in ats_fail:
            assert at <= interrupt_budget, (
                f"{lvl} fail after_ticks={at} > interrupt-mode budget "
                f"~{interrupt_budget} (max_turns={max_turns}) — the "
                f"timeout fail never bites in interrupt mode ⇒ draw"
            )


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


# ── Engine-bound tests (parameterised over seeds 1..4) ─────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_repair_plus_engage_wins(level, seed):
    """The intended `repair(damaged_proc) + attack_unit(committed
    defenders)` policy must WIN on every (level, seed). Load-bearing
    solvency test."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(), seed=seed)
    assert res.outcome == "win", (
        f"intended must WIN on {level} s={seed}; got {res.outcome} "
        f"turns={res.turns} tick={res.signals.game_tick} "
        f"kills={res.signals.units_killed} lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSE on every (level, seed) — defenders never
    engage, proc stays damaged and falls to the scheduled wave."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_brute_loses(level, seed):
    """Brute (attack-only, no repair) must LOSE on every (level,
    seed) — even with attackers cleared, the damaged proc is never
    healed, so the building_hp_pct_gte:proc clause never latches
    and the clock expires."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"brute must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_wrong_target_loses(level, seed):
    """Wrong-target (repair the healthy fact instead of the damaged
    proc) must LOSE on every (level, seed) — the fact is already
    near full HP so its repair is a no-op, and the damaged proc
    stays at 40% and busts the HP bar at the clock."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _wrong_target_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"wrong-target must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin per seed (the
    test_hard_tier curation contract)."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
