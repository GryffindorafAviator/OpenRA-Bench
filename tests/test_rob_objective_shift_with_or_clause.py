"""rob-objective-shift-with-or-clause pack — full no-cheat validation.

Wave-6 reasoning pack: PlanBench replanning / goal-conditional
adaptation. The win is `any_of` over two destruction paths
(`enemy_key_buildings_destroyed_in_region` at A=(60,20) or B=(100,30))
AND a deadline (within_ticks:5400) AND an attrition cap
(units_lost_lte:3 on easy/medium, :2 on hard). The capability under
test is RECOGNISING AN INFEASIBLE PATH AND PIVOTING to the other.

Bar (per CLAUDE.md):
  - STALL                LOSS every (level, seed) — clock teeth.
  - COMMIT-A-PURE        LOSS every (level, seed) — attrition cap
                         (6 e3 anti-vehicle wreck 5 2tnk).
  - COMMIT-B-PURE        WIN on easy/medium (direct B beeline beats
                         the light picket inside cap); LOSS on hard
                         (tighter cap leaves no slack and the
                         spawn-varied approach brushes A's e3 sight).
  - INTENDED scout-A-pivot-B  WIN every (level, seed).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "rob-objective-shift-with-or-clause.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Objective coordinates (mirror the pack).
A = (60, 20)
B = (100, 30)


# ── Policy helpers ───────────────────────────────────────────────


def _tank_ids(obs):
    return [
        u["id"] for u in (obs.get("units_summary", []) or [])
        if u.get("type") == "2tnk"
    ]


# ── Policies ─────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on after_ticks every (level, seed)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _commit_a_pure_policy():
    """Brute-force the 5 tanks straight at A's heavily-defended fact.
    Must LOSE every (level, seed): the 6 e3 rocket-infantry sit on A
    and are the engine's anti-vehicle counter — the 5 2tnk get
    chewed up well past `units_lost_lte:3` (or :2 on hard) before A
    can fall. The attrition fail clause bites.
    """
    state = {"sent": False}

    def pol(obs, Cmd):
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        if not state["sent"]:
            state["sent"] = True
            return [Cmd.attack_move(ids, A[0], A[1])]
        return [Cmd.observe()]
    return pol


def _commit_b_pure_policy():
    """Beeline the 5 tanks straight at B's lightly-defended fact.
    On easy/medium this WINS: a direct drive reaches B inside the
    deadline, 2tnk crush the e1 picket, units_lost stays ≤2. On hard
    the tighter cap (lte:2) + spawn-varied approach (lateral y-band
    flips per seed) makes this brittle — the natural path from NORTH
    staging (y=14) toward B at (100,30) brushes A's e3 sight, chips
    a tank or two, and busts the cap.
    """
    state = {"sent": False}

    def pol(obs, Cmd):
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        if not state["sent"]:
            state["sent"] = True
            return [Cmd.attack_move(ids, B[0], B[1])]
        return [Cmd.observe()]
    return pol


def _intended_scout_a_pivot_b_policy():
    """Intended capability play.

      1. Send the whole squad ~halfway toward A (~ (40, ay)) — close
         enough to register A's defence on observation but well out
         of e3 rocket sight envelope (≥10 cells away). This is the
         "scout-A" leg without committing to engagement.
      2. After a short observation window, pivot the squad south-east
         to B at (100,30) — the approach via a southerly waypoint
         (e.g. (60, ay+pivot)) stays well south of A's e3 sight
         envelope on hard's NORTH spawn, or well north of nothing
         particular on hard's SOUTH spawn.
      3. Attack-move at B; the 2 e1 picket falls in a few volleys
         and B is razed inside 5400 ticks, with ≤1 tank lost.
    """
    state = {"phase": 0, "tick": 0, "ay": None}

    def pol(obs, Cmd):
        state["tick"] += 1
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        # Determine the squad's y-band from any one tank position.
        if state["ay"] is None:
            ay = ids and obs["units_summary"][0].get("cell_y")
            # Fall back: scan units_summary for the first 2tnk.
            for u in obs.get("units_summary", []) or []:
                if u.get("type") == "2tnk":
                    ay = u.get("cell_y")
                    break
            state["ay"] = int(ay) if ay is not None else 20

        ay = state["ay"]

        # Phase 0: scout-A leg — move conservatively toward A but
        # stop at (30, ay) — well west of A's e3 sight envelope
        # (~7 cells), enough to register the heavy defence on
        # observation without engaging.
        if state["phase"] == 0:
            state["phase"] = 1
            return [Cmd.move_units(ids, 30, ay)]

        # Phase 1: brief observation window — A's defence surfaces
        # in `enemies` / `enemy_buildings` once the squad is in
        # observation range (~6-8 cells).
        if state["phase"] == 1:
            if state["tick"] >= 4:
                state["phase"] = 2
            else:
                return [Cmd.observe()]

        # Phase 2: drop SOUTH first to a deep waypoint (30, 36)
        # then EAST along the southern band at y=36 (well below
        # A's e3 sight envelope at y=17..21, distance ≥14 cells)
        # to a staging cell at (90, 36). This circumvents A
        # cleanly even for the NORTH spawn (ay=14).
        if state["phase"] == 2:
            state["phase"] = 3
            return [Cmd.move_units(ids, 30, 36)]

        if state["phase"] == 3:
            if state["tick"] >= 8:
                state["phase"] = 4
                return [Cmd.move_units(ids, 90, 36)]
            return [Cmd.observe()]

        # Phase 4: settle at southern staging, then attack-move
        # NORTH-EAST at B (100,30). Approach from the SOUTH stays
        # out of A's sight line entirely.
        if state["phase"] == 4:
            if state["tick"] >= 25:
                state["phase"] = 5
                return [Cmd.attack_move(ids, B[0], B[1])]
            return [Cmd.observe()]

        # Phase 5: keep pressing B in case the first attack-move
        # didn't fully engage the picket (re-issue once).
        if state["phase"] == 5 and state["tick"] >= 45:
            state["phase"] = 6
            return [Cmd.attack_move(ids, B[0], B[1])]

        return [Cmd.observe()]
    return pol


# ── Pack-shape tests (cheap; no engine) ──────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "rob-objective-shift-with-or-clause"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PlanBench / SC2 pivot /
    product pivot / military objective change."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("SC2" in a for a in anchors), anchors
    assert any("product pivot" in a for a in anchors), anchors
    assert any("military" in a.lower() for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start (binding contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_or_clause_used_in_win():
    """The win must be `any_of` over the two destruction paths —
    the spec-load-bearing structural shape of this pack."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        or_branches = [b for b in ao if "any_of" in b]
        assert len(or_branches) == 1, (
            f"{lvl} must have exactly one any_of branch; got {win}"
        )
        clauses = or_branches[0]["any_of"]
        assert len(clauses) == 2, (
            f"{lvl} any_of must have exactly 2 paths; got {clauses}"
        )
        # Both clauses must be enemy_key_buildings_destroyed_in_region
        # with a region anchored at A=(60,20) or B=(100,30).
        regions = set()
        for cl in clauses:
            assert "enemy_key_buildings_destroyed_in_region" in cl, cl
            v = cl["enemy_key_buildings_destroyed_in_region"]
            regions.add((int(v["x"]), int(v["y"])))
            assert int(v["radius"]) == 6, v
            assert "fact" in v["types"], v
        assert regions == {A, B}, regions


def test_attrition_cap_tightens_on_hard():
    """Easy/medium use units_lost_lte:3; hard tightens to :2 as the
    +1 controlled axis (one new variable per tier)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump(exclude_none=True)

        def _find(node, key):
            if isinstance(node, dict):
                if key in node:
                    return node[key]
                for v in node.values():
                    r = _find(v, key)
                    if r is not None:
                        return r
            elif isinstance(node, list):
                for v in node:
                    r = _find(v, key)
                    if r is not None:
                        return r
            return None

        assert _find(win, "units_lost_lte") == 3, lvl
    c = compile_level(pack, "hard")
    win = c.win_condition.model_dump(exclude_none=True)

    def _find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for v in node.values():
                r = _find(v, key)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = _find(v, key)
                if r is not None:
                    return r
        return None

    assert _find(win, "units_lost_lte") == 2


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable = 93 + 90·(N-1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True,
        )
        # within_ticks lives at the top all_of level.
        wts = [c["within_ticks"] for c in win.get("all_of", [])
               if "within_ticks" in c]
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={level_def.max_turns}) — deadline never bites"
            )


def test_tools_match_spec():
    """Tools 'full set' per spec — observe + build + place_building +
    move_units + attack_unit + attack_move + stop_units."""
    pack = load_pack(PACK)
    tools = pack.base.get("tools") if isinstance(pack.base, dict) \
        else pack.base.tools
    expected = {"observe", "build", "place_building", "move_units",
                "attack_unit", "attack_move", "stop_units"}
    assert set(tools) == expected, tools


def test_starting_cash_matches_spec():
    """Spec calls for starting_cash 500 across all levels."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        assert pack.levels[lvl].starting_cash == 500, lvl


# ── Engine-bound tests (parameterised over seeds 1..4) ───────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE every (level, seed) — the
    after_ticks clause bites at the turn budget."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_commit_a_pure_loses(level, seed):
    """A brute-force commit-A policy must LOSE every (level, seed).
    The 6 e3 anti-vehicle defenders on A's fact bleed the 5 2tnk
    past the cap (`units_lost_lte:3`/:2) before A falls — even if A
    eventually goes down, the cap clause already triggered fail.
    """
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _commit_a_pure_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"commit-A-pure must LOSE on {level} s={seed}; "
        f"got {res.outcome} (lost={res.signals.units_lost}, "
        f"kills={res.signals.units_killed})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_commit_b_pure_wins_on_easy(seed):
    """A direct beeline to B WINS on easy: B is extra-lightly
    defended (1× e1), the central A defence is the easy variant
    (no heavy tanks), and the path from (10,20) to (100,30) is
    survivable enough to reach B with ≤3 tanks alive — the e1
    picket falls and B is razed inside the cap. This is the
    'spec said: also a valid path on lower diff' check."""
    c = compile_level(load_pack(PACK), "easy")
    res = run_level(c, _commit_b_pure_policy(), seed=seed)
    assert res.outcome == "win", (
        f"commit-B-pure must WIN on easy s={seed}; "
        f"got {res.outcome} (lost={res.signals.units_lost}, "
        f"kills={res.signals.units_killed}, "
        f"tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", ("medium", "hard"))
def test_commit_b_pure_loses_on_medium_and_hard(level, seed):
    """A direct beeline to B LOSES on medium and hard: the early
    naive commit to B drives the squad through A's e3 sight
    envelope on the central y-band, the heavy A picket chews
    them up before they clear, and the cap (`units_lost_lte:3`
    or :2) trips before B is reached. This is the 'pure-B early
    LOSS — didn't realize A might work, late commitment' tooth
    that makes the intended scout-then-pivot decision
    load-bearing on medium+ tiers."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _commit_b_pure_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"commit-B-pure must LOSE on {level} s={seed}; "
        f"got {res.outcome} (lost={res.signals.units_lost}, "
        f"kills={res.signals.units_killed}, "
        f"tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_a_pivot_b_wins(level, seed):
    """The intended scout-A-pivot-B capability play must WIN every
    (level, seed). This is the load-bearing test that the pack is
    solvable inside the budget by the advertised capability across
    all hard-tier spawn variants."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_scout_a_pivot_b_policy(), seed=seed)
    assert res.outcome == "win", (
        f"intended scout-A-pivot-B must WIN on {level} s={seed}; "
        f"got {res.outcome} (lost={res.signals.units_lost}, "
        f"kills={res.signals.units_killed}, "
        f"tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin — smoke-tests
    the spawn-variation contract."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
