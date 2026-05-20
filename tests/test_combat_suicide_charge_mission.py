"""combat-suicide-charge-mission — sacrifice the force to destroy a
high-value enemy objective (forlorn hope / military sacrifice doctrine).

The bar: an all-in commit that drives the WHOLE force straight at the
enemy fact (110, 20), accepting total force loss, WINS on every level
and every hard seed (1..4). STALL (only observe) and PRESERVE-FORCE
(only the lead tank probes, the rest hold) both LOSE on every level —
non-win is a real reachable timeout LOSS via the `after_ticks` fail
clause (no own_units_gte:1 clause; total force loss is NOT a fail,
this is the whole point of the sacrifice anchor).

Validation is scripted (no model / network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-suicide-charge-mission.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-suicide-charge-mission"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) >= 3, (
        f"benchmark_anchor must list the sacrifice anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sacrifice", "forlorn", "sc2"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, lost=0, fact_destroyed=False,
         fact_xy=(110, 20)):
    """Synthesize a WinContext for predicate-level checks.

    `fact_destroyed=True` simulates the objective fact at `fact_xy`
    having been destroyed (the engine's `enemy_buildings_destroyed_
    records` carries the type + cell coords).
    """
    import types

    types_count = {"fact": 1} if fact_destroyed else {}
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        enemy_buildings_destroyed=1 if fact_destroyed else 0,
        enemy_buildings_destroyed_types=types_count,
        enemy_buildings_destroyed_records=(
            [("fact", int(fact_xy[0]), int(fact_xy[1]))]
            if fact_destroyed
            else []
        ),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _alive(n):
    return [
        {"cell_x": 10, "cell_y": 20, "type": "2tnk", "id": str(1000 + i)}
        for i in range(n)
    ]


def test_predicates_all_levels():
    """Win iff (objective fact razed) AND (within deadline).
    Force-wipe with the fact razed is STILL a WIN (the whole point
    of the sacrifice anchor — survival is NOT required).
    Fact intact + clock expired is a real fail."""
    for lvl, within in (("easy", 5400), ("medium", 6400), ("hard", 7200)):
        c = compile_level(load_pack(PACK_PATH), lvl)
        # Intended: fact destroyed in budget — WIN even with zero units left
        assert evaluate(
            c.win_condition,
            _ctx(units=[], tick=within - 100, lost=10, fact_destroyed=True),
        ), f"{lvl}: force-wipe-but-fact-razed must WIN"
        # Fact destroyed AND units alive — also a WIN (lower-loss path)
        assert evaluate(
            c.win_condition,
            _ctx(units=_alive(3), tick=within - 100, lost=7,
                 fact_destroyed=True),
        ), f"{lvl}: fact razed in budget must WIN regardless of survivors"
        # Fact NOT destroyed — never a win even with all units alive
        assert not evaluate(
            c.win_condition,
            _ctx(units=_alive(10), tick=within - 100, lost=0,
                 fact_destroyed=False),
        ), f"{lvl}: preserve-the-force without razing the fact must not win"
        # Past deadline — not a win even if fact razed late
        assert not evaluate(
            c.win_condition,
            _ctx(units=_alive(5), tick=within + 1, lost=5,
                 fact_destroyed=True),
        ), f"{lvl}: late raze must not win"
        # Timeout with fact intact → fail
        assert evaluate(
            c.fail_condition,
            _ctx(units=_alive(5), tick=within + 1, lost=5,
                 fact_destroyed=False),
        ), f"{lvl}: deadline expiry with fact intact must FAIL"
        # Force-wipe with fact intact — NOT an automatic fail
        # (only the deadline can fail; the model must rely on the
        # clock, not unit count). This is what makes "preserve" a
        # strict losing strategy: it can't run-down-the-clock OR
        # avoid death to win, only RAZE the fact.
        assert not evaluate(
            c.fail_condition,
            _ctx(units=[], tick=within - 100, lost=10,
                 fact_destroyed=False),
        ), (
            f"{lvl}: pre-deadline force-wipe must NOT be a fail "
            "(sacrifice anchor — survival is not required, only "
            "the objective)"
        )


def test_predicate_region_scoped_to_objective():
    """The destroyed `fact` must be at (110, 20) — a fact destroyed
    elsewhere (e.g. the far-corner SENTINEL fact at (122, 4)) does
    NOT satisfy the in-region predicate."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK_PATH), lvl)
        # Sentinel fact at (122, 4) razed (would only happen if the
        # agent burned ticks on the wrong target) — predicate must
        # NOT trigger.
        import types
        sig = types.SimpleNamespace(
            game_tick=100,
            units_killed=0,
            units_lost=0,
            cash=0,
            resources=0,
            own_buildings=[],
            own_building_types=set(),
            enemies_seen_ids=set(),
            enemy_buildings_seen_ids=set(),
            enemy_buildings_destroyed=1,
            enemy_buildings_destroyed_types={"fact": 1},
            enemy_buildings_destroyed_records=[("fact", 122, 4)],
        )
        ctx = WinContext(signals=sig, render_state={"units_summary": []})
        assert not evaluate(c.win_condition, ctx), (
            f"{lvl}: razing the sentinel fact at (122,4) must NOT "
            "satisfy the in-region predicate at (110,20)"
        )


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks must be reachable in max_turns
    (~90 ticks/turn; max tick ≈ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    expected = {"easy": 5401, "medium": 6401, "hard": 7201}
    for lvl, after in expected.items():
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert after <= max_tick, (
            f"{lvl}: after_ticks {after} > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the staging latitude (NORTH y=10 / SOUTH y=30).
    Engine-roundtrip is asserted by tests/test_hard_tier.py once the
    pack is added to UPGRADED."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, f"hard needs 2 spawn_point groups, got {groups}"


def test_objective_fact_and_sentinel_present():
    """Two enemy `fact` per level: one OBJECTIVE fact at (110, 20)
    and one SENTINEL fact far away (~(122, 4)). The sentinel keeps
    the episode alive past objective-fact destruction so the
    within_ticks predicate evaluates on the terminal frame (engine
    auto-done footgun on MustBeDestroyed enemy buildings —
    CLAUDE.md)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        facts = [
            (a.position[0], a.position[1])
            for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert len(facts) == 2, (
            f"{lvl}: must have 2 enemy facts (objective + sentinel), "
            f"got {len(facts)} at {facts}"
        )
        objective = [(x, y) for (x, y) in facts
                     if (x - 110) ** 2 + (y - 20) ** 2 <= 36]
        sentinel = [(x, y) for (x, y) in facts
                    if (x - 110) ** 2 + (y - 20) ** 2 > 36]
        assert len(objective) == 1 and len(sentinel) == 1, (
            f"{lvl}: expected 1 objective fact at ~(110,20) + 1 "
            f"sentinel far away, got obj={objective} sent={sentinel}"
        )


def test_actors_inside_playable_bounds():
    """rush-hour-arena playable bounds are (2..126, 2..38). Every
    authored actor must be inside or the engine panics."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} owner={a.owner} at {a.position} "
                f"is outside rush-hour-arena playable bounds"
            )


def test_no_force_wipe_fail_clause():
    """Critical sacrifice-anchor invariant: the fail clause MUST NOT
    contain `not own_units_gte:1` (force-wipe). If it does, the
    intended commit-all strategy fails the moment the last attacker
    dies — even if the fact has already been razed — which inverts
    the anchor. The only fail path is the deadline expiring."""
    import yaml
    raw = yaml.safe_load(PACK_PATH.read_text())
    for lvl in ("easy", "medium", "hard"):
        fail = raw["levels"][lvl].get("fail_condition") or {}

        def _has_force_wipe(node) -> bool:
            if isinstance(node, dict):
                if "not" in node:
                    inner = node["not"]
                    if isinstance(inner, dict) and "own_units_gte" in inner:
                        return True
                if "own_units_lte" in node and int(node["own_units_lte"]) == 0:
                    return True
                return any(_has_force_wipe(v) for v in node.values())
            if isinstance(node, list):
                return any(_has_force_wipe(v) for v in node)
            return False

        assert not _has_force_wipe(fail), (
            f"{lvl}: fail_condition must NOT include a force-wipe clause "
            "(sacrifice anchor — survival is NOT required); only the "
            f"deadline may fail. got {fail!r}"
        )


def test_termination_keeps_episode_alive_past_force_wipe():
    """Engine `agent_units_killed: false` is required: without it
    the engine auto-`done`s the run the moment the last attacker
    dies (the rush-hour-arena map carries the default Conquest
    rules), collapsing the win-by-objective into a draw — even if
    the objective fact has already been razed and within_ticks is
    still satisfied. This invariant must hold per level (overrides
    must NOT silently flip it back to True)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.scenario.termination.agent_units_killed is False, (
            f"{lvl}: termination.agent_units_killed must be False "
            "(sacrifice anchor — engine must not auto-done on "
            "force wipe before the objective evaluates)"
        )


def test_enemy_picket_anti_armor():
    """The defender must be lethal to a careful approach (e3 rocket
    infantry anti-tank Dragon, range 5, damage 5000 vs Heavy; +
    4tnk heavy mammoth tank). Without these, a "preserve the force"
    plan trivially wins — destroying the discriminator."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        enemy_types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert "e3" in enemy_types, (
            f"{lvl}: defender must include e3 rocket infantry (anti-armor)"
        )
        assert "4tnk" in enemy_types, (
            f"{lvl}: defender must include 4tnk heavy tank (close brawl)"
        )
        assert enemy_types.count("fact") == 2, (
            f"{lvl}: exactly 2 enemy fact (objective + sentinel)"
        )


# ── engine-driven scripted policies ─────────────────────────────────


def _stall(rs, Command):
    """Pure observe — fact never takes damage; clock runs out → LOSS."""
    return [Command.observe()]


def _all_in(rs, Command):
    """Commit-all: every agent unit drives straight at the objective
    fact at (110, 20). When any enemy comes into vision the whole
    force attack_unit's the closest threat, switching to the
    objective fact the moment it appears in vision. Tanks + rocket
    infantry absorb fire and trade themselves for damage on the
    fact. The intended sacrifice play."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    enemies = rs.get("enemy_summary", []) or []
    if enemies:
        # Prioritise the objective fact if in vision (cell_x < 115
        # filters out the far-corner sentinel fact at (122,4) which
        # sits past the objective).
        for e in enemies:
            t = (e.get("type") or "").lower()
            if t == "fact" and int(e.get("cell_x", 200)) < 115:
                return [Command.attack_unit(ids, str(e["id"]))]
        # Otherwise smash the nearest defender first.
        tx = sum(int(u["cell_x"]) for u in units) / len(units)
        ty = sum(int(u["cell_y"]) for u in units) / len(units)
        enemies.sort(
            key=lambda e: (int(e.get("cell_x", 0)) - tx) ** 2
            + (int(e.get("cell_y", 0)) - ty) ** 2
        )
        return [Command.attack_unit(ids, str(enemies[0]["id"]))]
    return [Command.attack_move(ids, 110, 20)]


def _preserve_force(rs, Command):
    """Preserve plan: only the LEAD tank pushes east while the rest
    hold the west staging — and any tank below 50% HP retreats back
    to the west edge. Saves units but dribbles damage onto the fact
    (the lead tank gets shredded long before the fact falls); clock
    runs out → LOSS. Discriminates careful-skirmish from the
    all-in commit."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    # Find the eastmost unit — that's the "lead probe"; everyone
    # else holds position.
    sorted_units = sorted(units, key=lambda u: -int(u.get("cell_x", 0)))
    lead = sorted_units[0]
    holders = sorted_units[1:]
    # Lead pushes east — but if damaged, retreats to safety.
    lead_hp = int(lead.get("health", lead.get("hp", 100)) or 100)
    if lead_hp < 50:
        cmds.append(Command.move_units([str(lead["id"])], 6, int(lead["cell_y"])))
    else:
        cmds.append(Command.attack_move([str(lead["id"])], 100, 20))
    # Holders explicitly stop and stay west.
    for h in holders:
        cmds.append(Command.stop([str(h["id"])]))
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_all_in_commit_wins(level, seed):
    """Intended sacrifice play: commit ALL units straight at the
    objective. Must WIN on every level and every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _all_in, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: all-in commit should WIN, got {r.outcome} "
        f"after {r.turns} turns "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    """Pure observe: fact never takes damage; clock runs out → LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(fact never takes damage), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_preserve_force_loses(level, seed):
    """Preserve-the-force / careful-skirmish: only a lead tank probes
    east while the rest hold and any damaged tank retreats. Damage
    on the fact dribbles; clock expires → LOSS. Discriminates
    careful from the intended all-in commit."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _preserve_force, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: preserve-force must LOSE "
        f"(insufficient damage on the fact), got {r.outcome} "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )
