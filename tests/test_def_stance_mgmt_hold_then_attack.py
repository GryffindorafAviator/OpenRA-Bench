"""def-stance-mgmt-hold-then-attack — stance-flip ambush trigger pack
(Wave-8 ACTION capability).

The bar (per CLAUDE.md): the intended (set_stance-then-hold) policy
WINS on every level and every hard seed (1..4); the stall policy
(observe only — never lift the stance from HoldFire) LOSES on every
level and every hard seed. Non-win is a real reachable timeout LOSS
(after_ticks 4501 fits inside max_turns 55 ⇒ 4501 ≤ 93 + 90·54 = 4953,
so the deadline bites as a real LOSS rather than collapsing to a
draw).

The discrimination is stance-management:
  • Intended: lift defenders from HoldFire (stance:0) to AttackAnything
    (stance:3) via `set_stance` so the engine's stance-driven auto-
    fire engages the inbound rusher band as it closes to cannon range
    at the choke; 4× 2tnk cannons one-shot e1 (combat-focus-fire-
    priority / CLAUDE.md) and wipe the column before it reaches the
    fact. ≥K kills, ≥3 of 4 tanks alive, fact intact, well under
    the tick budget ⇒ WIN.
  • Stall (only observe): defenders stay on stance:0 the entire run.
    The rusher band overruns the silent tank cluster (the tanks die
    without firing back), then turns on the fact and razes it ⇒
    `not has_building:fact` LOSS. Kills stay at 0 (`units_killed_gte`
    also fails) so the win predicate is busted on multiple clauses
    every level / seed.

NOTE on the flip-immediately variant: in this combat cell with
stationary 2tnks on stance:3 the engine's auto-targeting fires the
instant any enemy enters cannon range — there is no behavioural
difference between "flip on turn 1" and "flip when the rush is in
the choke" because the tanks do not pursue beyond their starting
cluster (def-pre-position-mobile-reserve engine fact). The load-
bearing capability is therefore the `set_stance` verb itself; the
timing axis is encoded in the level brief as a doctrinal note but
the win predicate measures only the EFFECT (kills + survival +
fact intact). A model that always flips on turn 1 WINS — that is
the correct outcome: the verb was called, the engine did the rest.
A model that never flips LOSES — the verb was the discriminator.

Validation is scripted (no model / network) — these policies are
exhaustive proxies for the stance-management capability and exercise
the predicate teeth (has_building / units_killed_gte / own_units_gte
/ within_ticks) directly through eval_core.run_level. See CLAUDE.md
"How to validate" for the harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "def-stance-mgmt-hold-then-attack.yaml"

LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


# ── structural / metadata checks (no engine) ──────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "def-stance-mgmt-hold-then-attack"
    assert pack.meta.capability == "action"
    # benchmark_anchor must name the ROE + SC2 stance + ambush anchors
    # the pack is built against (suite-enforced by
    # test_benchmark_anchor_required, but spot-checked here too).
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors
    joined = " ".join(anchors).lower()
    assert "roe" in joined or "rules-of-engagement" in joined or "rules of engagement" in joined
    assert "sc2" in joined or "stance" in joined
    assert "ambush" in joined
    # meta required-prose fields populated and on-theme.
    rwm = pack.meta.real_world_meaning.lower()
    assert "stance" in rwm or "hold" in rwm or "set_stance" in rwm
    assert "ambush" in rwm or "trigger" in rwm or "overwatch" in rwm
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_allowlist_includes_set_stance():
    """The base `tools:` allowlist must include `set_stance` — it is
    the verb under test (without it the agent cannot lift defenders
    from HoldFire and the scenario is unsolvable)."""
    pack = load_pack(PACK_PATH)
    base = pack.base if isinstance(pack.base, dict) else pack.base.dict()
    tools = set(base.get("tools") or [])
    assert "set_stance" in tools, (
        "set_stance MUST be on the allowlist — it is the agent's only "
        "way to lift the pre-staged defenders from HoldFire and the "
        "load-bearing verb the pack tests"
    )
    assert "observe" in tools


def test_defenders_start_on_holdfire_at_the_choke():
    """The defenders (2tnk) on every level must be pre-placed at the
    choke on stance:0 (HoldFire). If they start on any other stance
    the scenario degenerates (a stall WINS because auto-engage fires
    without the agent doing anything — the verb is no longer load-
    bearing)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tanks = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert tanks, f"{lvl}: no 2tnk defenders found"
        for t in tanks:
            stance = getattr(t, "stance", None)
            assert stance == 0, (
                f"{lvl}: defender 2tnk at {t.position} starts on stance "
                f"{stance}, expected 0 (HoldFire) — otherwise the "
                f"set_stance verb is not load-bearing (a stall policy "
                f"would win by auto-engagement)"
            )


def test_each_level_has_a_reachable_timeout_fail():
    """Non-win must be a real LOSS, not a draw. The `after_ticks` fail
    must be strictly below the tick reachable at max_turns (≤90 ticks/
    turn ⇒ tick ≤ 93 + 90·(max_turns-1))."""
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
            f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
        )
        # within_ticks in the win clause must ALSO be reachable (else
        # a passing intended policy times out as a draw too).
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
    """The win clause MUST be of the form
    `all_of[ has_building:fact, units_killed_gte:K, own_units_gte:3,
    within_ticks:T ]` — matches the Wave-8 scenario brief and ensures
    the four load-bearing axes (asset survival + engagement + force
    preservation + clock) are all measured."""
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
    """The fail clause MUST be `any_of[ after_ticks:T+1,
    not has_building:fact, not own_units_gte:1 ]` — every non-win
    pathway (timeout, base lost, force wiped) is a real LOSS."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        flat = str(fc)
        assert "after_ticks" in flat, (lvl, fc)
        assert "has_building" in flat and "fact" in flat, (lvl, fc)
        assert "own_units_gte" in flat, (lvl, fc)


def test_hard_defines_two_agent_spawn_point_groups():
    """Hard-tier contract (UPGRADED): ≥2 distinct seed-driven agent
    spawn_point groups so the choke latitude rotates by seed (anti-
    memorisation of an absolute trigger cell — the doctrine must
    generalise across either spawn)."""
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
    # In-bounds check (rush-hour-arena playable y ≈ 2..38).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_pack_is_in_upgraded_not_in_not_applicable():
    """Round-trip check: this pack IS spawn-varied on hard, so it must
    appear in UPGRADED and NOT in NOT_APPLICABLE."""
    from tests.test_hard_tier import NOT_APPLICABLE, UPGRADED

    assert "def-stance-mgmt-hold-then-attack" in UPGRADED, (
        "pack defines ≥2 spawn groups on hard; must appear in UPGRADED"
    )
    assert "def-stance-mgmt-hold-then-attack" not in NOT_APPLICABLE, (
        "pack must not be in NOT_APPLICABLE — it IS spawn-varied"
    )


# ── engine-driven scripted policies (intended WINS, stall LOSES) ────


def _intended_policy(rs, Command):
    """Lift defenders from HoldFire (stance:0) to AttackAnything
    (stance:3) every turn. The tank cluster's auto-targeting then
    engages the rusher band the instant it closes to cannon range
    (the tanks don't pursue beyond the choke — def-pre-position-
    mobile-reserve engine fact — so the salvo happens at the choke
    regardless of which turn the flip is issued). The agent never
    issues attack_unit / attack_move; the kills are the engine's
    stance-driven auto-fire."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if u.get("type") == "2tnk"
    ]
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    return [Command.set_stance(ids, 3), Command.observe()]


def _stall_policy(rs, Command):
    """Issue only observe(); never lift the stance, never move. The
    defenders stay on stance:0 (HoldFire) — they never engage; the
    rusher band overruns the silent cluster (tanks die without
    firing), then turns on the fact and razes it ⇒ `not has_building:
    fact` LOSS. Kills stay at 0 (independent fail on units_killed_gte)
    so the win predicate is busted on multiple clauses."""
    return [Command.observe()]


@pytest.mark.parametrize("level", LEVELS)
def test_intended_policy_wins_every_level_and_seed(level):
    """Intended set_stance(3) wins on every level; on hard, across all
    4 seeds (UPGRADED contract)."""
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
    """Pure-observe stall: defenders stay on HoldFire, rusher overruns
    the cluster, fact razed ⇒ LOSS on every level / seed. Non-win
    must be a real LOSS, not a draw — `after_ticks 4501` reaches
    within `max_turns 55` (≤93 + 90·54 = 4953)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (fact razed or force "
            f"wiped), got {res.outcome} "
            f"(killed={res.signals.units_killed}, "
            f"lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick})"
        )
        # Stall must rack up exactly 0 kills (defenders never fire on
        # HoldFire) — this is the proof that `set_stance` is the load-
        # bearing verb (a "kill the rush via auto-engage" play is
        # impossible without lifting the stance).
        assert res.signals.units_killed == 0, (
            f"{level} seed={s}: stall (no set_stance call) killed "
            f"{res.signals.units_killed} units — defenders should be "
            f"silent on HoldFire (verb is not load-bearing if kills > 0)"
        )


def test_intended_run_is_deterministic_on_easy():
    """Same seed must reproduce the same outcome + turn count — the
    bench's seed-based reproducibility contract."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "easy")
    a = run_level(c, _intended_policy, seed=2)
    b = run_level(c, _intended_policy, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
