"""combat-pincer-coordination — synchronised two-prong pincer attack.

The bar: TWO 3-tank squads start on the SAME west edge at OPPOSING
latitudes (north y=8, south y=32). A central enemy cluster sits at
the mid-map (around (50,20)). The win predicate REQUIRES both squads
to engage simultaneously:

  • `units_in_region_gte:{x:50,y:20,radius:8,n:4}` — at least 4 tanks
    must be inside the central region; a single squad has only 3 tanks
    so single-squad-A cannot satisfy this clause regardless of
    attrition.
  • `units_lost_lte:2` — sequenced (A first, B late) is shredded by
    the cluster's anti-armour mass and loses ≥3 tanks before B
    arrives → busts the cap.
  • `within_ticks:4500` — stall hits the deadline → after_ticks LOSS.

Discrimination (the four-script bar; the scripted runs cover all
seeds 1–4 per level on every policy):

  • stall                  — every level/seed must LOSE (clock).
  • single-squad-A         — every level/seed must LOSE (region n=4).
  • sequenced-A-then-B-late — medium/hard must LOSE (attrition cap
                              busted by lone Squad A); easy is the
                              bare-skill tier where this may squeak
                              by (SCENARIO_REVIEW_CHECKLIST inert-
                              easy-teeth convention).
  • intended pincer-sync   — every level/seed must WIN.

Validation is scripted (no model / network)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-pincer-coordination.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-pincer-coordination"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "pincer", "envelopment", "synchronisation"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, kills=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        power_provided=0,
        power_drained=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        explored_percent=0.0,
        then_progress={},
        seq_progress={},
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _tanks_at(n, x=50, y=20):
    return [
        {"cell_x": x, "cell_y": y, "type": "2tnk", "id": str(2000 + i)}
        for i in range(n)
    ]


def test_predicates_per_level():
    """Region n=4 + kill bar + attrition + deadline all load-bearing."""
    pack = load_pack(PACK_PATH)
    expectations = {
        # (kill_bar, region_n, lost_cap, within, after_fail)
        "easy":   (6, 4, 2, 4500, 4501),
        "medium": (8, 4, 2, 4500, 4501),
        "hard":   (8, 4, 2, 4500, 4501),
    }
    for lvl, (kb, n, cap, w, af) in expectations.items():
        c = compile_level(pack, lvl)
        # Intended win: ≥n tanks in region, ≥kb kills, ≤cap lost, in time.
        ctx_win = _ctx(units=_tanks_at(n + 1), tick=w - 100, kills=kb, lost=cap)
        assert evaluate(c.win_condition, ctx_win), f"{lvl}: intended should WIN"
        # Region clause unmet (only 3 tanks at objective) → not a win
        # — this is the load-bearing pincer teeth (single squad has only 3 tanks).
        ctx_3 = _ctx(units=_tanks_at(3), tick=w - 100, kills=kb, lost=0)
        assert not evaluate(c.win_condition, ctx_3), (
            f"{lvl}: 3 tanks in region < n={n} must not satisfy win"
        )
        # Kill bar unmet → not a win.
        ctx_low_kills = _ctx(units=_tanks_at(n + 1), tick=w - 100, kills=kb - 1, lost=0)
        assert not evaluate(c.win_condition, ctx_low_kills), (
            f"{lvl}: kills < {kb} must not satisfy win"
        )
        # Attrition cap busted → not a win + fails.
        ctx_lost = _ctx(units=_tanks_at(n + 1), tick=w - 100, kills=kb, lost=cap + 1)
        assert not evaluate(c.win_condition, ctx_lost)
        assert evaluate(c.fail_condition, ctx_lost), f"{lvl}: lost>{cap} must FAIL"
        # Timeout busted → fails.
        ctx_late = _ctx(units=_tanks_at(n + 1), tick=af + 1, kills=kb, lost=0)
        assert evaluate(c.fail_condition, ctx_late), f"{lvl}: tick>after must FAIL"
        # Force-wipe → fails.
        ctx_wipe = _ctx(units=[], tick=w - 100, kills=kb, lost=6)
        assert evaluate(c.fail_condition, ctx_wipe)


def test_timeout_reachable_inside_max_turns():
    """No DRAW degeneracy: after_ticks fail trigger must be reachable
    within max_turns (engine advances ~90 ticks per turn ⇒ max tick
    ≈ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        # After-ticks fail clause is 4501 on every level here.
        assert 4501 <= max_tick, (
            f"{lvl}: after_ticks 4501 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )


def test_two_squads_on_every_level():
    """Two 3-tank squads at opposing latitudes (north y≈8 vs south
    y≈32); the pincer mechanism requires both prongs."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
        # All agent units are medium tanks (2tnk).
        for a in agent_actors:
            assert a.type == "2tnk", (
                f"{lvl}: agent must be 2tnk (medium tanks); got {a.type}"
            )
        # Two latitudes represented (north y<20 and south y>20).
        latitudes = {("N" if a.position[1] < 20 else "S") for a in agent_actors}
        assert latitudes == {"N", "S"}, (
            f"{lvl}: agent squads must span both latitudes; got {latitudes}"
        )


def test_enemy_cluster_avoids_silent_no_place_cell():
    """CLAUDE.md: (50,20) silently fails to place enemy clusters.
    Cluster must anchor on the off-by-one cells (50,19) / (50,21)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        cluster = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type in ("e1", "e3", "2tnk")
        ]
        assert cluster, f"{lvl}: no central cluster declared"
        for a in cluster:
            assert tuple(a.position) != (50, 20), (
                f"{lvl}: enemy cluster member at (50,20) — CLAUDE.md "
                f"silent-no-place footgun"
            )
            assert a.position[0] == 50 and a.position[1] in (19, 21), (
                f"{lvl}: cluster member off-anchor; got {a.position}"
            )


def test_persistent_far_enemy_marker_present():
    """Engine auto-`done` mitigation: a far unarmed enemy `fact` keeps
    the episode alive past full cluster elimination so within_ticks /
    region predicates evaluate on the terminal frame (no DRAW collapse)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        far_facts = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
            and a.position[0] >= 100  # far east of the cluster
        ]
        assert far_facts, f"{lvl}: needs a far persistent enemy fact sentinel"


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins which squad ID is north vs south."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


# ── engine-driven scripted policies ─────────────────────────────────


def _split_squads(rs):
    """Return (north_ids, south_ids) — split agent 2tnk units by
    latitude. The agent's two staging latitudes are y≈8 (north) and
    y≈32 (south); the cluster sits at y=20."""
    n_ids, s_ids = [], []
    for u in (rs.get("units_summary") or []):
        if str(u.get("type", "")).lower() != "2tnk":
            continue
        if int(u["cell_y"]) < 20:
            n_ids.append(str(u["id"]))
        else:
            s_ids.append(str(u["id"]))
    return n_ids, s_ids


def _stall(rs, Command):
    """Pure observe — agent units never engage → kill bar unmet AND
    region clause unmet → after_ticks LOSS."""
    return [Command.observe()]


def _single_squad_a(rs, Command):
    """Send only the NORTH squad; hold the SOUTH squad. Even if A
    clears the cluster on its own, only 3 tanks can be in the
    objective region → `units_in_region_gte:{n:4}` cannot be
    satisfied → LOSS. On medium/hard the cluster's enemy 2tnks also
    shred A → units_lost_lte:2 busts as a second teeth."""
    a, b = _split_squads(rs)
    cmds = []
    if a:
        cmds.append(Command.attack_move(a, 50, 20))
    if b:
        cmds.append(Command.stop(b))
    return cmds or [Command.observe()]


def _make_sequenced_a_then_b_late(delay_turns=20):
    """Squad A commits first; Squad B holds for `delay_turns` then
    advances. The cluster shreds A in detail before B arrives →
    units_lost_lte:2 busts on medium/hard. On easy with no enemy
    armour A may clear the cluster cleanly and B walks in → may WIN
    (acceptable per the inert-easy-teeth convention)."""
    state = {"turn": 0}

    def policy(rs, Command):
        state["turn"] += 1
        a, b = _split_squads(rs)
        cmds = []
        if a:
            cmds.append(Command.attack_move(a, 50, 20))
        if b:
            if state["turn"] >= delay_turns:
                cmds.append(Command.attack_move(b, 50, 20))
            else:
                cmds.append(Command.stop(b))
        return cmds or [Command.observe()]

    return policy


def _intended_pincer_sync(rs, Command):
    """Both squads attack_move the cluster on turn 1. The two squads
    are equidistant from the cluster (~45 cells each) so a naive
    simultaneous launch arrives together; mass DPS clears the
    cluster in ~6s before either side bleeds out."""
    a, b = _split_squads(rs)
    cmds = []
    if a:
        cmds.append(Command.attack_move(a, 50, 20))
    if b:
        cmds.append(Command.attack_move(b, 50, 20))
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(no engagement → kill bar + region clause unmet), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_single_squad_a_loses(level, seed):
    """Single-squad-A cannot satisfy `units_in_region_gte:{n:4}` —
    only 3 tanks exist in Squad A so the region clause is structurally
    unsatisfiable. On medium/hard the cluster also shreds A as a
    second teeth (units_lost_lte:2 busts)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _single_squad_a, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: single-squad-A must LOSE "
        f"(region n=4 impossible with 3 tanks); got {r.outcome} "
        f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_sequenced_a_then_b_late_loses_on_medium_hard(level, seed):
    """A commits first, B holds 20 turns. The cluster's enemy 2tnks
    shred A (lose 3 of 3 tanks) before B arrives → units_lost_lte:2
    busts → LOSS. Easy has no enemy armour so this may squeak by
    there (bare-skill-tier inert-easy-teeth convention)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _make_sequenced_a_then_b_late(20), seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: sequenced-A-then-B-late must LOSE "
        f"(A shredded by enemy 2tnks before B arrives); got {r.outcome} "
        f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_pincer_sync_wins(level, seed):
    """Both squads attack_move the cluster on turn 1 — naive
    simultaneous launch with equidistant staging arrives together,
    mass DPS clears the cluster before either side bleeds out."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended_pincer_sync, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended pincer-sync should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
    )
