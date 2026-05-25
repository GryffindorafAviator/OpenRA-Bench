"""combat-stance-mgmt-attack — stance-flip hunt-authorisation pack
(Wave-9 ACTION capability, ROE escalation / SC2 stance micro
anchor).

The bar (per CLAUDE.md): the intended (set_stance(3)) policy WINS
on every level and every hard seed (1..4); the stall policy
(observe only — never escalate from stance:1 ReturnFire) LOSES on
every level and every hard seed. Non-win is a real reachable
timeout LOSS (after_ticks 4501 fits inside max_turns 60 ⇒ 4501 ≤
93 + 90·59 = 5403, so the deadline bites as a real LOSS rather
than collapsing to a draw).

The discrimination is hunt-authorisation:
  • Intended: escalate the 4× 2tnk formation from ReturnFire
    (stance:1) to AttackAnything (stance:3) via `set_stance`. The
    engine's stance:3 auto-engage scan finds the nearest visible
    enemy in sight+hunt-bonus and advances the tank toward it;
    on arrival the next-tick scan promotes the encounter into
    an Attack. 2tnk cannon vs e1 ≈ 1-shot, vs 1tnk ≈ 6 shots —
    all comfortably under the tick budget. ≥K kills, ≥3 of 4
    tanks alive (no incoming damage — the scatter is stance:0),
    fact intact (the scatter never closes on the agent's fact),
    well under the deadline ⇒ WIN.
  • Stall (only observe): the formation stays on stance:1
    (ReturnFire). The scattered enemies are stance:0 (never fire
    first), so the tanks never receive fire ⇒ return-fire gate
    never opens ⇒ tanks never advance ⇒ kills stay at 0.
    `units_killed_gte:K` fails; the after_ticks deadline fires
    ⇒ LOSS.

Validation is scripted (no model / network); these policies are
exhaustive proxies for the hunt-authorisation capability and
exercise the predicate teeth (has_building / units_killed_gte /
own_units_gte / within_ticks) directly through eval_core.run_level.

This pack ships alongside an engine fix that makes stance:1 truly
return-fire-only and stance:3 a real hunting stance — see
`OpenRA-Rust/openra-sim/tests/test_stance_semantics.rs` for the
pinning tests on the engine side.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-stance-mgmt-attack.yaml"

LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


# ── structural / metadata checks (no engine) ──────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-stance-mgmt-attack"
    assert pack.meta.capability == "action"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors
    joined = " ".join(anchors).lower()
    assert "roe" in joined or "rules-of-engagement" in joined or "escalation" in joined
    assert "sc2" in joined or "stance" in joined
    rwm = pack.meta.real_world_meaning.lower()
    assert "stance" in rwm or "escalat" in rwm or "set_stance" in rwm
    assert "hunt" in rwm or "pursue" in rwm or "advance" in rwm
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_allowlist_includes_set_stance():
    """`set_stance` is the load-bearing verb — without it the agent
    cannot escalate the formation from ReturnFire to AttackAnything
    and the pack is unsolvable."""
    pack = load_pack(PACK_PATH)
    base = pack.base if isinstance(pack.base, dict) else pack.base.dict()
    tools = set(base.get("tools") or [])
    assert "set_stance" in tools
    assert "observe" in tools


def test_allowlist_excludes_offensive_verbs():
    """F1 cheat-WIN audit fix (2026-05-25): `move_units`,
    `attack_unit`, `attack_move` are NOT in the toolset. With them the
    agent could brute attack_unit on each scattered enemy and clear
    the kill bar without ever calling `set_stance` — making the verb-
    under-test cosmetic. Restricting the toolset to {observe, stop,
    set_stance} forces every winning play to escalate the stance.
    """
    pack = load_pack(PACK_PATH)
    base = pack.base if isinstance(pack.base, dict) else pack.base.dict()
    tools = set(base.get("tools") or [])
    for forbidden in ("move_units", "attack_unit", "attack_move"):
        assert forbidden not in tools, (
            f"{forbidden!r} re-enabled — the brute attack_unit chain "
            f"that the F1 audit caught will WIN again. Keep the "
            f"toolset locked to stance-only verbs."
        )


def test_defenders_start_on_returnfire_at_the_west_edge():
    """The 4× 2tnk formation on every level must start on stance:1
    (ReturnFire). If they start on any other stance the scenario
    degenerates — stance:0 inverts to the HOLD pack (a different
    capability); stance:2 / stance:3 auto-engage from t=0 and a
    stall would win by accident."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tanks = [a for a in c.scenario.actors if a.owner == "agent" and a.type == "2tnk"]
        assert tanks, f"{lvl}: no 2tnk defenders"
        for t in tanks:
            stance = getattr(t, "stance", None)
            assert stance == 1, (
                f"{lvl}: defender 2tnk at {t.position} starts on stance "
                f"{stance}, expected 1 (ReturnFire) — otherwise the "
                f"set_stance(3) verb is not load-bearing"
            )


def test_scattered_enemies_are_passive():
    """The scattered enemy force MUST be on stance:0 — if they fire
    first, the ReturnFire gate opens automatically and the tanks
    engage without the agent calling set_stance. That would make
    the stall policy WIN by accident."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        # The enemy fact is fine on any stance (it doesn't fire);
        # only check armed scatter units.
        ENEMY_COMBAT = {"e1", "e3", "1tnk", "2tnk", "3tnk", "jeep"}
        for a in c.scenario.actors:
            if a.owner != "enemy" or a.type not in ENEMY_COMBAT:
                continue
            stance = getattr(a, "stance", None)
            assert stance == 0, (
                f"{lvl}: scattered enemy {a.type} at {a.position} on "
                f"stance {stance}, expected 0 (passive) — an aggressive "
                f"enemy unlocks return-fire and breaks the discriminator"
            )


def test_each_level_has_a_reachable_timeout_fail():
    """Non-win must be a real LOSS, not a draw."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        deadline = None
        for clause in fc.get("any_of", []) or []:
            if "after_ticks" in clause:
                deadline = int(clause["after_ticks"])
        assert deadline is not None, f"{lvl}: no after_ticks fail clause"
        reachable = 93 + 90 * (c.max_turns - 1)
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
    """Win = all_of[ has_building:fact, units_killed_gte:K,
    own_units_gte:3, within_ticks:T ]."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        flat = str(wc)
        assert "has_building" in flat and "fact" in flat, (lvl, wc)
        assert "units_killed_gte" in flat, (lvl, wc)
        assert "own_units_gte" in flat, (lvl, wc)
        assert "within_ticks" in flat, (lvl, wc)


def test_fail_predicate_carries_the_task_idiom():
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        flat = str(fc)
        assert "after_ticks" in flat
        assert "has_building" in flat and "fact" in flat
        assert "own_units_gte" in flat


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

    assert "combat-stance-mgmt-attack" in UPGRADED
    assert "combat-stance-mgmt-attack" not in NOT_APPLICABLE


# ── engine-driven scripted policies (intended WINS, stall LOSES) ────


def _intended_policy(rs, Command):
    """Escalate the formation from ReturnFire (stance:1) to
    AttackAnything (stance:3) every turn. The engine's stance:3
    hunt path advances each tank toward the nearest visible enemy;
    on arrival the in-range branch one-shots e1 and 6-shots 1tnk.
    The agent never issues attack_unit / attack_move / move_units;
    the kills are pure stance-driven auto-fire + hunt."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    return [Command.set_stance(ids, 3), Command.observe()]


def _stall_policy(rs, Command):
    """Issue only observe(); never escalate the stance. The
    formation stays on ReturnFire forever; the scatter is stance:0
    so no incoming fire ever lands ⇒ return-fire gate never opens
    ⇒ tanks never advance ⇒ kills stay at 0 ⇒
    `units_killed_gte:K` fails ⇒ after_ticks LOSS."""
    return [Command.observe()]


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
            f"lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick}, "
            f"buildings={res.signals.own_buildings})"
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
            f"lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick})"
        )
        # Stall must score exactly 0 kills — formation never fires
        # because (a) ReturnFire requires incoming damage that the
        # passive scatter never delivers, and (b) the hunt path is
        # only engaged at stance:3.
        assert res.signals.units_killed == 0, (
            f"{level} seed={s}: stall killed {res.signals.units_killed} "
            f"units — the formation should be silent on stance:1 "
            f"against a passive scatter (verb is not load-bearing if "
            f"kills > 0)"
        )


def test_intended_run_is_deterministic_on_easy():
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "easy")
    a = run_level(c, _intended_policy, seed=2)
    b = run_level(c, _intended_policy, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns)
