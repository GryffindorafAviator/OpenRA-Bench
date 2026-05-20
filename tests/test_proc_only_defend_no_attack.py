"""proc-only-defend-no-attack — defensive ROE under a strict no-attack
tool ban (Wave-6 Group I procedural-compliance pack).

The bar (per CLAUDE.md): the intended (set_stance-then-hold) policy
WINS on every level and every hard seed (1..4); the brute (attack_unit
— a FORBIDDEN tool), stall (only observe), and brute-attack-move
(attack_move) policies LOSE on every level and every hard seed. Non-win
is a real reachable timeout LOSS (after_ticks 4501 fits inside
max_turns 55 ⇒ 4501 ≤ 93 + 90·54 = 4953).

The discrimination is procedural compliance:
  • Intended: lift defenders from HoldFire via `set_stance` so the
    engine's stance-driven auto-fire racks up ≥2 (easy/medium) or ≥3
    (hard) kills as the patrol probes oscillate into range; the agent
    NEVER calls an attack tool.
  • Brute (attack_unit / attack_move): `tool_violations_gte:1` fires
    on the first turn that issues the forbidden command — LOSS.
  • Stall (only observe): stance:0 holds, defenders never engage, the
    probe chips the fact to death ⇒ `not has_building:fact` LOSS (and
    independently `units_killed_gte:2` is never satisfied).

Validation is scripted (no model / network) — these policies are
exhaustive proxies for the procedural success / failure modes and
exercise the predicate teeth (has_building / units_killed_gte /
within_ticks / tool_violations_gte) directly through
eval_core.run_level. See CLAUDE.md "How to validate" for the harness.
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
PACK_PATH = PACKS / "proc-only-defend-no-attack.yaml"


# ── unit-level predicate / schema checks (no engine) ──────────────────


def _ctx(
    units_xy=(),
    tick=1000,
    killed=0,
    lost=0,
    tool_violations=0,
    own_buildings=(),
):
    """Synthesize a WinContext for predicate-level checks. Mirrors the
    field set the live signals object exposes during eval_core."""
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        tool_violations=tool_violations,
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "proc-only-defend-no-attack"
    assert pack.meta.capability == "action"
    # benchmark_anchor must name the IFBench + ROE + BFCL anchors the
    # pack is built against (suite-enforced by
    # test_benchmark_anchor_required, but spot-checked here too).
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors
    joined = " ".join(anchors).lower()
    assert "ifbench" in joined or "negative" in joined
    assert "roe" in joined or "rules-of-engagement" in joined or "rules of engagement" in joined
    assert "bfcl" in joined or "allowlist" in joined
    assert "guard" in joined or "defensive" in joined
    # meta required-prose fields populated
    assert (
        "rules-of-engagement" in pack.meta.real_world_meaning.lower()
        or "rules of engagement" in pack.meta.real_world_meaning.lower()
    )
    assert (
        "guard" in pack.meta.robotics_analogue.lower()
        or "security" in pack.meta.robotics_analogue.lower()
    )
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None
        # Every level must carry the procedural-compliance allowlist:
        # attack_unit / attack_move are FORBIDDEN.
        assert "attack_unit" in c.forbidden_tools
        assert "attack_move" in c.forbidden_tools


def test_allowlist_includes_set_stance_and_excludes_attacks():
    """The base `tools:` allowlist must include `set_stance` (the verb
    that lifts defenders from HoldFire and is the agent's whole
    permitted offensive lever) and MUST NOT include either attack
    tool — the two are listed only in `forbidden_tools` per level."""
    pack = load_pack(PACK_PATH)
    base = pack.base if isinstance(pack.base, dict) else pack.base.dict()
    tools = set(base.get("tools") or [])
    assert "set_stance" in tools, (
        "set_stance must be on the allowlist — it is the agent's "
        "ONLY permitted offensive lever (defenders auto-fire under "
        "stance, not under attack_unit)"
    )
    assert "move_units" in tools
    assert "observe" in tools
    assert "attack_unit" not in tools
    assert "attack_move" not in tools


def test_each_level_has_tool_violations_fail_clause():
    """The fail clause MUST carry `tool_violations_gte: 1` so a single
    forbidden call ends the episode as a LOSS — zero-tolerance ROE."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Keep fact alive + ≥1 unit so non-violation clauses don't
        # confound; vary only the tool_violations counter.
        units = [(14, 18), (14, 20)]
        bldgs = [("fact", 16, 20)]
        ctx0 = _ctx(units_xy=units, tick=100, tool_violations=0, own_buildings=bldgs)
        ctx1 = _ctx(units_xy=units, tick=100, tool_violations=1, own_buildings=bldgs)
        assert not evaluate(c.fail_condition, ctx0), (
            f"{lvl}: clean turn must not fail"
        )
        assert evaluate(c.fail_condition, ctx1), (
            f"{lvl}: a single forbidden call must fail the episode"
        )


def test_predicates_per_level_kill_target_and_deadline():
    pack = load_pack(PACK_PATH)
    expected_kills = {"easy": 2, "medium": 2, "hard": 3}
    units = [(14, 18), (14, 20), (14, 22)]
    bldgs = [("fact", 16, 20)]
    for lvl, k in expected_kills.items():
        c = compile_level(pack, lvl)
        # Intended: k kills, fact alive, tick under deadline → WIN.
        assert evaluate(
            c.win_condition,
            _ctx(units_xy=units, tick=3000, killed=k, own_buildings=bldgs),
        ), f"{lvl}: ≥{k} kills + fact alive + in-time must satisfy the win"
        # One kill short → not yet a win.
        assert not evaluate(
            c.win_condition,
            _ctx(units_xy=units, tick=3000, killed=k - 1, own_buildings=bldgs),
        ), f"{lvl}: {k-1} kills (target {k}) must NOT yet win"
        # Fact razed → win impossible AND fail clause fires.
        assert not evaluate(
            c.win_condition,
            _ctx(units_xy=units, tick=3000, killed=k, own_buildings=[]),
        ), f"{lvl}: fact razed must NOT win even with ≥{k} kills"
        assert evaluate(
            c.fail_condition,
            _ctx(units_xy=units, tick=3000, killed=k, own_buildings=[]),
        ), f"{lvl}: fact razed must fail"
        # Past deadline → real loss, reachable within max_turns.
        assert evaluate(
            c.fail_condition,
            _ctx(units_xy=units, tick=4502, killed=k, own_buildings=bldgs),
        ), f"{lvl}: past after_ticks must fail (reachable inside max_turns)"
        # No draw degeneracy: deadline must fit inside max_turns.
        assert 4501 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 4501 must be reachable within max_turns "
            f"(else timeout draws instead of losing); "
            f"max_turns={c.max_turns}"
        )


def test_hard_defines_two_agent_spawn_point_groups():
    """Hard tier must define ≥2 agent spawn_point groups (UPGRADED
    contract); the win/fail predicates are symmetric so either staging
    latitude (NORTH y=17..19 or SOUTH y=21..23) can win."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert groups == {0, 1}, (
        f"hard must define ≥2 agent spawn_point groups for seed-driven "
        f"start variation; got {sorted(groups)}"
    )


def test_pack_is_in_upgraded_not_in_not_applicable():
    """Round-trip check: this pack IS spawn-varied on hard, so it must
    appear in UPGRADED and NOT in NOT_APPLICABLE."""
    from tests.test_hard_tier import NOT_APPLICABLE, UPGRADED

    assert "proc-only-defend-no-attack" in UPGRADED, (
        "pack defines ≥2 spawn groups on hard; must appear in UPGRADED"
    )
    assert "proc-only-defend-no-attack" not in NOT_APPLICABLE, (
        "pack must not be in NOT_APPLICABLE — it IS spawn-varied"
    )


# ── engine-driven scripted policies (4 policies × 1/1/4 seeds) ──────


def _intended_policy(rs, Command):
    """Lift defenders from HoldFire to AttackAnything (stance=3) once,
    then idle. The patrol probes oscillate into rifle range; the
    engine's stance-driven auto-fire racks up the required kills. The
    agent never issues attack_unit / attack_move (zero violations)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    # set_stance(units, 3) on every turn — idempotent and ensures the
    # lift survives any engine quirks across resets/turns. observe()
    # keeps the action list non-empty even when the cmd is a no-op.
    return [Command.set_stance(ids, 3), Command.observe()]


def _stall_policy(rs, Command):
    """Issue only observe(); never lift the stance, never move. The
    defenders stay on stance:0 (HoldFire) — they never engage; the
    patrol probes chip the fact to death ⇒ `not has_building:fact`
    LOSS (and independently `units_killed_gte:2` is never met)."""
    return [Command.observe()]


def _brute_attack_unit_policy(rs, Command):
    """Brute issues a forbidden attack_unit on EVERY turn against a
    stub target id (the bench tracks the command NAME before the engine
    evaluates the order — see tests/test_forbidden_tools.py). The
    tool_violations counter rises on turn 1 ⇒ `tool_violations_gte:1`
    fail clause fires ⇒ LOSS regardless of tactical outcome."""
    return [Command.observe(), Command.attack_unit(["1"], target_id="99999")]


def _brute_attack_move_policy(rs, Command):
    """The sibling brute path: attack_move (also forbidden) on every
    turn. Same fail mode as attack_unit — single forbidden call ⇒ LOSS.
    Distinct policy because the spec calls out BOTH tools as forbidden;
    the test must verify EITHER instantly trips the counter."""
    return [Command.observe(), Command.attack_move(["1"], target_x=30, target_y=20)]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_policy_wins(level):
    """Intended set_stance-then-hold wins on every level; on hard,
    across all 4 seeds (UPGRADED contract)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended set_stance policy must WIN, "
            f"got {res.outcome} after {res.turns} turns "
            f"(killed={res.signals.units_killed}, "
            f"lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick}, "
            f"viol={res.signals.tool_violations})"
        )
        # No spurious tool violations from the intended (stance-only) policy.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: intended policy issued forbidden tool "
            f"{res.signals.tool_violations} time(s); should be 0"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Pure-observe stall: defenders stay on HoldFire, probes chip fact
    OR the deadline expires with 0 kills ⇒ LOSS on every level / seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (fact razed or deadline), "
            f"got {res.outcome} "
            f"(killed={res.signals.units_killed}, "
            f"tick={res.signals.game_tick})"
        )
        # Stall must not produce any tool violations — its failure mode
        # is the deadline / fact loss, NOT procedural compliance.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: stall must not violate; "
            f"violations were {res.signals.tool_violations}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_unit_policy_loses_on_tool_violation(level):
    """Brute issues a forbidden attack_unit each turn → tool_violations
    accumulates → `tool_violations_gte:1` fires → LOSS. Holds on every
    level and every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_attack_unit_policy, seed=s)
        assert res.signals.tool_violations >= 1, (
            f"{level} seed={s}: brute attack_unit must trip ≥1 violation; "
            f"got {res.signals.tool_violations} "
            f"(tools_called={res.signals.tools_called})"
        )
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute attack_unit must LOSE via "
            f"tool_violations_gte:1; got {res.outcome}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_move_policy_loses_on_tool_violation(level):
    """Sibling brute path — attack_move is also forbidden; the counter
    must fire on it too. Same LOSS-on-turn-1 mode."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_attack_move_policy, seed=s)
        assert res.signals.tool_violations >= 1, (
            f"{level} seed={s}: brute attack_move must trip ≥1 violation; "
            f"got {res.signals.tool_violations} "
            f"(tools_called={res.signals.tools_called})"
        )
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute attack_move must LOSE via "
            f"tool_violations_gte:1; got {res.outcome}"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: `after_ticks: 4501` must fit inside max_turns
    on every level (~90 ticks/turn ⇒ tick ≤ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl
