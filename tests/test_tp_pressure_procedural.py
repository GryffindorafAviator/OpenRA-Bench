"""tp-pressure-procedural — procedural compliance under TIME pressure
(Wave-10 ACTION capability; IFBench-under-stress / SOC runbook
compliance / BFCL V4 anchor).

This pack is the INTERSECTION of two sibling mechanics:
  • combat-stance-mgmt-attack — the stance:3 hunt-authorisation
    verb (escalate a stance:1 ReturnFire formation to stance:3
    AttackAnything so the engine's hunt path clears a passive
    enemy band). On that pack attack_unit/attack_move are ALLOWED.
  • strict-toolban-fidelity-under-pressure — `forbidden_tools` +
    the `tool_violations` counter + a tight clock.

Here `attack_unit` and `attack_move` are FORBIDDEN, the clock is
tight, and the ONLY sanctioned way to score the kills is the
set_stance(3) hunt.

The bar (per CLAUDE.md) — holds on every level and every hard
seed (1..4):
  • intended (set_stance(ids, 3) then let the hunt run) WINS:
    ≥4 kills, fact intact, zero tool violations, comfortably
    under the tight within_ticks deadline.
  • stall (only observe()) LOSES: the formation stays on stance:1
    ReturnFire; the band is stance:0 (never fires first) so the
    return-fire gate never opens ⇒ 0 kills ⇒ units_killed_gte:4
    fails ⇒ the after_ticks deadline bites ⇒ a real reachable
    timeout LOSS (not a draw — the inert enemy fact marker keeps
    the episode alive past band-elim).
  • brute (attack_unit / attack_move) LOSES: the bench's
    tool_violations counter increments BEFORE the engine
    evaluates the order ⇒ tool_violations_gte:1 fires ⇒ instant
    LOSS. Even an attack_unit on a non-existent id counts.

Validation is scripted (no model / no network); these policies
are exhaustive proxies for the capability and exercise the
predicate teeth (units_killed_gte / building_count_gte /
within_ticks / tool_violations_gte) directly via
eval_core.run_level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "tp-pressure-procedural.yaml"

LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


# ── structural / metadata checks (no engine) ──────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "tp-pressure-procedural"
    assert pack.meta.capability == "action"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors
    joined = " ".join(anchors).lower()
    assert "ifbench" in joined
    assert "soc" in joined or "runbook" in joined
    assert "bfcl" in joined
    rwm = pack.meta.real_world_meaning.lower()
    assert "set_stance" in rwm or "stance" in rwm
    assert "forbidden" in rwm or "attack_move" in rwm or "attack_unit" in rwm
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_allowlist_is_movement_and_stance_only():
    """The tool allowlist is `[move_units, set_stance, stop,
    observe]`. attack_unit / attack_move must NOT be in `tools:` —
    they are forbidden; set_stance is the load-bearing verb."""
    pack = load_pack(PACK_PATH)
    base = pack.base if isinstance(pack.base, dict) else pack.base.dict()
    tools = set(base.get("tools") or [])
    assert tools == {"move_units", "set_stance", "stop", "observe"}, tools


def test_attack_tools_are_forbidden_on_every_level():
    """`forbidden_tools` must list attack_unit AND attack_move on
    every level — the bench's tool_violations counter is the binding
    procedural-compliance measurement."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        forbidden = {str(t).lower() for t in (c.forbidden_tools or [])}
        assert "attack_unit" in forbidden, (lvl, forbidden)
        assert "attack_move" in forbidden, (lvl, forbidden)


def test_formation_starts_on_returnfire():
    """The 4× 2tnk formation on every level must start on stance:1
    (ReturnFire). stance:0 inverts to a hold drill; stance:2/3
    auto-engage from t=0 and a staller would win by accident."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tanks = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert tanks, f"{lvl}: no 2tnk formation"
        for t in tanks:
            assert getattr(t, "stance", None) == 1, (
                f"{lvl}: 2tnk at {t.position} starts on stance "
                f"{getattr(t, 'stance', None)}, expected 1 (ReturnFire)"
            )


def test_enemy_band_is_passive():
    """The band MUST be stance:0 — if it fires first, the
    return-fire gate opens automatically and a staller wins by
    accident."""
    pack = load_pack(PACK_PATH)
    ENEMY_COMBAT = {"e1", "e3", "1tnk", "2tnk", "3tnk", "jeep"}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            if a.owner != "enemy" or a.type not in ENEMY_COMBAT:
                continue
            assert getattr(a, "stance", None) == 0, (
                f"{lvl}: enemy {a.type} at {a.position} on stance "
                f"{getattr(a, 'stance', None)}, expected 0 (passive)"
            )


def test_each_level_has_a_reachable_timeout_fail():
    """Non-win must be a real LOSS, not a draw: the after_ticks fail
    and the within_ticks win must both be reachable inside
    max_turns (tick ≤ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        reachable = 93 + 90 * (c.max_turns - 1)
        fc = c.fail_condition.model_dump(exclude_none=True)
        deadline = None
        for clause in fc.get("any_of", []) or []:
            if "after_ticks" in clause:
                deadline = int(clause["after_ticks"])
        assert deadline is not None, f"{lvl}: no after_ticks fail clause"
        assert deadline < reachable, (
            f"{lvl}: deadline {deadline} unreachable within "
            f"{c.max_turns} turns (max tick {reachable})"
        )
        wc = c.win_condition.model_dump(exclude_none=True)
        deadline_win = None
        for clause in wc.get("all_of", []) or []:
            if "within_ticks" in clause:
                deadline_win = int(clause["within_ticks"])
        assert deadline_win is not None, f"{lvl}: no within_ticks win clause"
        assert deadline_win < reachable, (
            f"{lvl}: win deadline {deadline_win} unreachable within "
            f"{c.max_turns} turns"
        )


def test_win_predicate_carries_the_task_idiom():
    """Win = all_of[ units_killed_gte:K, building_count_gte:fact:1,
    within_ticks:T, not tool_violations_gte:1 ]."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        flat = str(c.win_condition.model_dump(exclude_none=True))
        assert "units_killed_gte" in flat, (lvl, flat)
        assert "building_count_gte" in flat and "fact" in flat, (lvl, flat)
        assert "within_ticks" in flat, (lvl, flat)
        # tool_violations_lte:0 is encoded as `not tool_violations_gte:1`.
        assert "tool_violations_gte" in flat, (lvl, flat)


def test_fail_predicate_carries_the_task_idiom():
    """Fail = any_of[ after_ticks:T+1, tool_violations_gte:1,
    not building_count_gte:fact:1 ]."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        flat = str(c.fail_condition.model_dump(exclude_none=True))
        assert "after_ticks" in flat, (lvl, flat)
        assert "tool_violations_gte" in flat, (lvl, flat)
        assert "building_count_gte" in flat and "fact" in flat, (lvl, flat)


def test_hard_defines_two_agent_spawn_point_groups():
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert groups == {0, 1}, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(groups)}"
    )
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_pack_is_in_upgraded_not_in_not_applicable():
    from tests.test_hard_tier import NOT_APPLICABLE, UPGRADED

    assert "tp-pressure-procedural" in UPGRADED
    assert "tp-pressure-procedural" not in NOT_APPLICABLE


# ── engine-driven scripted policies ───────────────────────────────────


def _intended_policy(rs, Command):
    """Escalate the formation from ReturnFire (stance:1) to
    AttackAnything (stance:3). The engine's stance:3 hunt path
    advances each tank to the band and the in-range branch scores
    the kills. The agent never invokes a forbidden tool — the kills
    are pure stance-driven auto-fire + hunt."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    return [Command.set_stance(ids, 3), Command.observe()]


def _stall_policy(rs, Command):
    """Only observe(); never escalate. The formation stays on
    stance:1; the passive band never fires ⇒ 0 kills ⇒ deadline
    LOSS."""
    return [Command.observe()]


def _brute_attack_move_policy(rs, Command):
    """attack_move toward the band — the obvious local-solve under
    the tight clock. attack_move is forbidden ⇒ instant
    tool_violations LOSS."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    return [Command.attack_move(ids, target_x=64, target_y=20)]


def _brute_attack_unit_policy(rs, Command):
    """attack_unit on a non-existent target id — still counts as a
    forbidden-tool invocation (the counter increments BEFORE the
    engine evaluates the order)."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    return [Command.attack_unit(ids, "nonexistent-target-id-999")]


@pytest.mark.parametrize("level", LEVELS)
def test_intended_policy_wins_every_level_and_seed(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended set_stance(3) policy must WIN, "
            f"got {res.outcome} after {res.turns} turns "
            f"(killed={res.signals.units_killed}, "
            f"violations={res.signals.tool_violations}, "
            f"tick={res.signals.game_tick}, "
            f"buildings={res.signals.own_buildings})"
        )
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: intended policy must not invoke any "
            f"forbidden tool; got {res.signals.tool_violations}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_stall_policy_loses_every_level_and_seed(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (no escalation ⇒ no "
            f"hunt ⇒ no kills ⇒ deadline bites), got {res.outcome} "
            f"(killed={res.signals.units_killed}, "
            f"tick={res.signals.game_tick})"
        )
        assert res.signals.units_killed == 0, (
            f"{level} seed={s}: stall killed "
            f"{res.signals.units_killed} units — the formation should "
            f"be silent on stance:1 against a passive band"
        )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy", [_brute_attack_move_policy, _brute_attack_unit_policy]
)
def test_brute_attack_policy_loses_every_level_and_seed(level, policy):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute {policy.__name__} must LOSE on a "
            f"forbidden-tool violation, got {res.outcome}"
        )
        assert res.signals.tool_violations >= 1, (
            f"{level} seed={s}: brute {policy.__name__} must trip the "
            f"tool_violations counter; got {res.signals.tool_violations}"
        )


def test_intended_run_is_deterministic_on_easy():
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "easy")
    a = run_level(c, _intended_policy, seed=2)
    b = run_level(c, _intended_policy, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns)
