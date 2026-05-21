"""def-while-building scenario family, full loop on Rust.

The pack tests CONCURRENT defence + construction under active attack:
while a heavy enemy rush is already pressuring the base, the agent
must SIMULTANEOUSLY queue new defensive pillboxes AND command the
pre-placed HoldFire (stance:0) tank defenders to attack the
incoming rush. Pausing either stream loses; only a play that
parallelises both wins.

* `building_count_gte:{pbox, n:3}` ⇒ the agent built the defensive
  pillbox bar (3 on every level — cash is intentionally tight at
  exactly 1800 = 3 × 600 with no slack);
* `building_count_gte:{type:fact,n:1}` ⇒ the fact must STAND at
  the deadline (PRESENT-TENSE — `has_building:fact` is a one-shot
  ever-seen set that stays true after the fact is razed, per
  CLAUDE.md footgun);
* `units_killed_gte:K` ⇒ the defence actually engaged the rush
  (HoldFire tanks that are never commanded sit idle and don't
  satisfy the bar);
* `within_ticks:T` paired with `after_ticks:T+1` ⇒ a non-finisher
  is a real reachable timeout LOSS (no interrupts on this pack ⇒
  each step is exactly 90 ticks, so max_turns is a hard tick
  budget the `after_ticks` deadline reliably bites in).

The scripted-policy validations prove deterministically that:

* the intended CONCURRENT policy (queue pbox + attack_unit tanks
  on the heavy 3tnk every turn from t=0) WINS every level +
  every hard seed (1..4);
* stall / build-only (queue pbox but never command tanks — the
  anti-infantry pillbox cannot kill the rush's heavy 3tnk, which
  raze the fact un-countered) / defend-only (attack_unit tanks
  but never build pbox — the pbox count bar is never met ⇒
  `within_ticks` unmet ⇒ clock LOSS) ALL LOSE every level +
  every hard seed — a real LOSS, not a draw;
* the hard tier defines ≥2 spawn_point groups (north fact y=14 /
  south fact y=26) so a memorised opening cannot generalise.

Recalibration note (engine movement+combat fixes): with the
pbox-fires + moving-units-take-fire changes, an all-infantry rush
was fully countered by the 3-pillbox cluster alone, so a
BUILD-ONLY play that never commanded the HoldFire tanks won for
free (the "laziest play wins" inversion). The rush now carries
two armoured heavy tanks (`3tnk`); the anti-infantry pillbox
(`M60mg`) cannot kill them in time, so stopping the armour
genuinely requires commanding the agent's medium tanks (`2tnk`,
anti-armour). The concurrent-streams capability is restored as
structurally load-bearing — the discrimination no longer hangs
on a knife-edge build-vs-rush timing race.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-while-building.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends and never moves. The
    pre-placed 2tnk are HoldFire (stance:0) so they do NOT engage
    on their own; the rush walks to the fact and razes it."""
    return [C.observe()]


def _own_pbox_count(rs):
    own_b = rs.get("own_buildings") or []
    return sum(1 for b in own_b if b.get("type") == "pbox")


def _own_fact(rs):
    own_b = rs.get("own_buildings") or []
    for b in own_b:
        if b.get("type") == "fact":
            return b
    return None


def _own_2tnk_ids(rs):
    units = rs.get("units_summary") or []
    return [u.get("id") for u in units if u.get("type") == "2tnk"]


def _visible_enemy_units(rs):
    return [e for e in (rs.get("enemy_summary") or []) if not e.get("is_building")]


def _focus_target(rs):
    """Focus the heavy tank (3tnk) first — the armoured 3tnk is the
    threat the anti-infantry pillbox cannot kill, so the medium
    tanks MUST clear it. Fall back to the nearest enemy if no 3tnk
    is visible."""
    enemies = _visible_enemy_units(rs)
    if not enemies:
        return None
    heavies = [e for e in enemies if e.get("type") == "3tnk"]
    return heavies[0] if heavies else enemies[0]


def _queue_and_place(rs, C, target_cells):
    """Common build-place loop: every turn queue another `pbox`
    (the Defense queue tolerates a backlog) and emit the next
    place at the cell matching the current pbox count."""
    pbox_count = _own_pbox_count(rs)
    cmds = []
    if pbox_count >= len(target_cells):
        return cmds
    cmds.append(C.build("pbox"))
    cell = target_cells[pbox_count]
    cmds.append(C.place_building("pbox", cell[0], cell[1]))
    return cmds


def make_intended_concurrent():
    """Intended: every turn (a) queue pbox + place at the next
    cell of a 3-cell cluster around the active fact AND (b)
    attack_unit the tanks at the heavy tank (3tnk) — the threat
    the anti-infantry pillbox cannot kill. Both streams advance
    from t=0; the pbox cluster comes online to mop the infantry
    while the commanded medium tanks kill the heavy armour. The
    cluster cells are read from the OBSERVED fact so the cluster
    centre flips with the spawn-driven hard latitude."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            fact = _own_fact(rs)
            if fact is None:
                return [C.observe()]
            fy = int(fact.get("cell_y", 20))
            # 3 cells clustering BEHIND the tank line (between the
            # tanks at x=14 and the fact at x=10): two flanking
            # cells at x=12 and a back cell at x=9 — all inside
            # the radius-3 disc around the fact (so a future
            # region predicate could anchor on this same shape).
            state["cells"] = [(12, fy - 1), (12, fy + 1), (9, fy + 2)]
        cmds = _queue_and_place(rs, C, state["cells"])
        tnks = _own_2tnk_ids(rs)
        tgt = _focus_target(rs)
        if tnks and tgt is not None:
            cmds.append(C.attack_unit(tnks, tgt.get("id")))
        if not cmds:
            cmds = [C.observe()]
        return cmds

    return policy


def make_build_only():
    """BUILD-ONLY: queue + place pbox every turn but NEVER command
    the tanks. The HoldFire tanks stay idle even when shot at;
    the anti-infantry pillbox cluster mops the rush's e1 but
    cannot kill the two armoured 3tnk, which raze the fact
    un-countered. LOSS via the fact-alive fail clause."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            fact = _own_fact(rs)
            if fact is None:
                return [C.observe()]
            fy = int(fact.get("cell_y", 20))
            state["cells"] = [(12, fy - 1), (12, fy + 1), (9, fy + 2)]
        cmds = _queue_and_place(rs, C, state["cells"])
        if not cmds:
            cmds = [C.observe()]
        return cmds

    return policy


def make_defend_only():
    """DEFEND-ONLY: attack_unit the tanks at the heavy 3tnk every
    turn but NEVER call build('pbox'). The commanded tanks kill
    the heavy armour and hold the fact, but the pbox count bar
    (≥3) is never satisfied ⇒ `within_ticks` unmet ⇒
    `after_ticks` fires → real timeout LOSS."""
    def policy(rs, C):
        cmds = []
        tnks = _own_2tnk_ids(rs)
        tgt = _focus_target(rs)
        if tnks and tgt is not None:
            cmds.append(C.attack_unit(tnks, tgt.get("id")))
        if not cmds:
            cmds = [C.observe()]
        return cmds

    return policy


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-while-building"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (SRE simultaneous incident-
    # response / business continuity / ops triage during incident).
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("sre" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("business continuity" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("ops triage" in a for a in anchors), pack.meta.benchmark_anchor
    # rusher bot wired through.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_is_exact_three_pbox_budget():
    """Cash is intentionally tight (3 pbox at 600 each = 1800, zero
    slack). A model that wastes cash on extras cannot complete the
    pbox count clause."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.starting_cash == 1800, (lvl, c.starting_cash)


def test_tank_defenders_are_holdfire_stance_zero():
    """The 2tnk defenders are pre-placed on stance:0 (HoldFire)
    every level. This is the load-bearing knob: a HoldFire tank
    does NOT engage on its own — it requires an explicit
    attack_unit / attack_move from the agent. Without HoldFire,
    `build-only` would coincidentally win (return-fire tanks
    auto-defend the fact)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        tnks = [a for a in c.scenario.actors
                if a.owner == "agent" and a.type == "2tnk"]
        assert tnks, f"{lvl}: no 2tnk defenders pre-placed"
        for a in tnks:
            assert a.stance == 0, (
                f"{lvl}: 2tnk @ {a.position} stance must be 0 "
                f"(HoldFire); got {a.stance}"
            )


def test_rush_carries_load_bearing_heavy_tanks():
    """The rush must include heavy tanks (3tnk) on every level.
    The 3tnk are the load-bearing element: the anti-infantry
    pillbox the agent builds cannot kill them, so stopping the
    rush genuinely requires commanding the agent's medium tanks.
    Without the 3tnk, a build-only play (pillbox cluster vs an
    all-infantry rush) would win for free."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        heavies = [a for a in c.scenario.actors
                   if a.owner == "enemy" and a.type == "3tnk"]
        # hard duplicates nothing — a single band on the symmetry
        # axis; easy/medium one band. Expect ≥2 heavy tanks.
        assert len(heavies) >= 2, (
            f"{lvl}: rush must carry ≥2 heavy 3tnk (the load-bearing "
            f"un-pbox-counterable threat); got {len(heavies)}"
        )
        for a in heavies:
            assert a.stance == 3, (
                f"{lvl}: rush 3tnk must be stance:3 (AttackAnything), "
                f"got {a.stance}"
            )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be strictly below the tick reachable at max_turns. No interrupts
    ⇒ each step is exactly 90 ticks (max tick = 93+90·(N-1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert deadline < reachable, (
        f"{level}: deadline {deadline} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


def test_fact_alive_clause_uses_present_tense_predicate():
    """The fact-survival clause must use the PRESENT-TENSE predicate
    (`building_count_gte:{type:fact,n:1}`) rather than `has_building`
    (a one-shot ever-seen set — CLAUDE.md footgun)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        fact_clauses = [
            clause for clause in fc.get("any_of", []) or []
            if isinstance(clause, dict)
            and isinstance(clause.get("not"), dict)
            and "building_count_gte" in (clause["not"] or {})
            and (clause["not"]["building_count_gte"] or {}).get("type") == "fact"
        ]
        assert fact_clauses, f"{lvl}: missing present-tense fact-alive fail clause"


def test_win_requires_pbox_count_and_kill_floor_and_deadline():
    """The win predicate must encode all three load-bearing clauses:
    pbox count (construction stream), kill floor (defence stream
    actually engaged), and within_ticks (concurrent — both streams
    finish before the deadline)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        clauses = win.get("all_of") or []
        keys = [list(cl.keys())[0] for cl in clauses if isinstance(cl, dict)]
        assert "building_count_gte" in keys, (lvl, keys)
        assert "units_killed_gte" in keys, (lvl, keys)
        assert "within_ticks" in keys, (lvl, keys)


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so the
    fact (and therefore the active defence latitude) flips by seed."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    fact_ys = sorted({
        a.position[1] for a in c.scenario.actors
        if a.owner == "agent" and a.type == "fact"
    })
    assert fact_ys == [14, 26], fact_ys
    # In-bounds check (rush-hour-arena playable x ≈ 2..126, y ≈ 2..38).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended CONCURRENT wins every level + every hard seed ─


@pytest.mark.parametrize("level", LEVELS)
def test_intended_concurrent_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended_concurrent(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended concurrent (queue pbox + "
            f"attack_unit tanks on the rocket soldier every turn) "
            f"must WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every single-stream policy LOSES (not draws) ───────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall",        lambda: stall),
        ("build_only",   make_build_only),
        ("defend_only",  make_defend_only),
    ],
)
def test_single_stream_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (HoldFire tanks idle, fact razed), build-only
    (anti-infantry pbox cannot kill the heavy 3tnk → fact razed),
    defend-only (pbox count bar never met → clock LOSS) — ALL
    must LOSE every level + every seed, no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended_concurrent(), seed=3)
    b = run_level(c, make_intended_concurrent(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
