"""coord-squad-handoff: sequenced squad handoff via the Wave-2 `then:`
happened-before composite + a new type-filtered region predicate
(`units_of_type_in_region_gte`).

The pack tests the SQUAD HANDOFF capability: Squad A (jeeps) must
deliver objective P1 FIRST, THEN Squad B (medium tanks) must deliver
P2; harder tiers add more alternating handoffs. The bar (per CLAUDE.md)
must hold on every level × every hard seed (1..4):

  - STALL          -> LOSS (clock)
  - B-FIRST        -> LOSS (then-latch never advances past A)
  - SINGLE-SQUAD   -> LOSS (the wrong-type squad can't satisfy clause 2)
  - INTENDED PLAY  -> WIN  (parallel A→P1, B→P2 with B holding for A's
                            latch)

A separate unit test pins the new predicate (`units_of_type_in_region
_gte`) to its exact semantics — it must NOT count units of the wrong
type at the region.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "coord-squad-handoff.yaml"
)

# Per-level expected handoff count: easy 2 (A→B); medium 3 (A→B→A);
# hard 4 (A→B→A→B).
EXPECTED_CLAUSES = {"easy": 2, "medium": 3, "hard": 4}


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


def _then_clauses(c):
    for cl in _win_clauses(c):
        if "then" in cl:
            return cl["then"]["clauses"]
    return None


# ── A. STRUCTURAL: predicate / order / deadline are wired correctly ──

@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_uses_then_with_type_filtered_region_clauses(level):
    pack = load_pack(PACK)
    assert pack.meta.capability == "action"
    c = compile_level(pack, level)
    clauses = _then_clauses(c)
    assert clauses is not None, f"{level}: must use a `then:` composite"
    assert len(clauses) == EXPECTED_CLAUSES[level], (
        f"{level}: expected {EXPECTED_CLAUSES[level]} handoff clauses, "
        f"got {len(clauses)}"
    )
    # Every clause is a type-filtered region predicate (the new
    # `units_of_type_in_region_gte`). Type-agnostic `units_in_region
    # _gte` would NOT enforce squad identity.
    for i, cl in enumerate(clauses):
        assert "units_of_type_in_region_gte" in cl, (
            f"{level} clause {i}: handoff must enforce squad identity"
        )
        v = cl["units_of_type_in_region_gte"]
        assert v["n"] >= 3, f"{level} clause {i}: needs n>=3 (full squad)"
        assert v["type"] in {"jeep", "2tnk"}, (
            f"{level} clause {i}: unknown squad unit type {v['type']}"
        )
    # Clauses must alternate squad types — that IS the handoff.
    types = [cl["units_of_type_in_region_gte"]["type"] for cl in clauses]
    for i in range(1, len(types)):
        assert types[i] != types[i - 1], (
            f"{level}: handoff clauses must alternate squad types, got {types}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_has_binding_deadline_and_real_loss(level):
    c = compile_level(load_pack(PACK), level)
    win = _win_clauses(c)
    wt = [cl["within_ticks"] for cl in win if "within_ticks" in cl]
    assert wt, f"{level}: missing within_ticks deadline"
    # The deadline must bite within max_turns (engine ~90 ticks/turn).
    assert wt[0] < 93 + 90 * (c.max_turns - 1), (
        f"{level}: within_ticks {wt[0]} unreachable inside max_turns "
        f"{c.max_turns} (would draw on timeout)"
    )
    # Every level must be able to emit a LOSS on timeout.
    fail = _fail_clauses(c)
    assert any("after_ticks" in cl for cl in fail), (
        f"{level}: missing after_ticks in fail_condition"
    )


def test_handoff_count_scales_with_difficulty():
    pack = load_pack(PACK)
    counts = [
        len(_then_clauses(compile_level(pack, lv)))
        for lv in ("easy", "medium", "hard")
    ]
    assert counts == [2, 3, 4], (
        f"handoff count should ladder 2->3->4, got {counts}"
    )


# ── B. PREDICATE UNIT TEST: units_of_type_in_region_gte semantics ────

class _Sig:
    def __init__(self):
        self.game_tick = 100
        self.then_progress: dict = {}


def _ctx(units):
    return WinContext(signals=_Sig(), render_state={"units_summary": units})


def test_type_filtered_region_predicate_ignores_wrong_type():
    """The whole point of the new predicate: 3 jeeps at P1 do NOT
    satisfy a `2tnk, n=3` clause at P1 — the wrong-type squad cannot
    take over for the right squad."""
    spec = {"units_of_type_in_region_gte":
        {"type": "2tnk", "x": 90, "y": 30, "radius": 8, "n": 3}}
    # 3 jeeps at the region — wrong type.
    jeeps = [
        {"type": "jeep", "cell_x": 90, "cell_y": 30, "id": str(i)}
        for i in range(3)
    ]
    assert evaluate(spec, _ctx(jeeps)) is False
    # Add 3 tanks (correct type) at the region — now satisfied.
    tanks = jeeps + [
        {"type": "2tnk", "cell_x": 91, "cell_y": 31, "id": "t%d" % i}
        for i in range(3)
    ]
    assert evaluate(spec, _ctx(tanks)) is True


def test_type_filtered_region_predicate_respects_radius():
    spec = {"units_of_type_in_region_gte":
        {"type": "jeep", "x": 50, "y": 10, "radius": 5, "n": 2}}
    near = [
        {"type": "jeep", "cell_x": 51, "cell_y": 11, "id": "1"},
        {"type": "jeep", "cell_x": 49, "cell_y": 10, "id": "2"},
    ]
    far = [{"type": "jeep", "cell_x": 60, "cell_y": 25, "id": "3"}]
    assert evaluate(spec, _ctx(near)) is True
    assert evaluate(spec, _ctx(far)) is False
    assert evaluate(spec, _ctx(near + far)) is True


def test_then_with_type_region_orders_squad_handoff():
    """End-to-end: a `then:` chain of type-filtered region clauses
    enforces the squad order — B-first never latches."""
    spec = {"then": {
        "id": "handoff",
        "clauses": [
            {"units_of_type_in_region_gte":
                {"type": "jeep", "x": 50, "y": 10, "radius": 8, "n": 3}},
            {"units_of_type_in_region_gte":
                {"type": "2tnk", "x": 90, "y": 30, "radius": 8, "n": 3}},
        ],
    }}
    sig = _Sig()
    # B-first: 3 tanks at P2, 0 jeeps at P1.
    b_first = [
        {"type": "2tnk", "cell_x": 90, "cell_y": 30, "id": "t%d" % i}
        for i in range(3)
    ]
    ctx = WinContext(signals=sig, render_state={"units_summary": b_first})
    assert evaluate(spec, ctx) is False
    assert sig.then_progress["handoff"] == 0  # B alone never advances A
    # Now A also arrives at P1 — chain completes in one evaluation
    # (clause 0 advances, clause 1 already satisfied).
    both = b_first + [
        {"type": "jeep", "cell_x": 50, "cell_y": 10, "id": "j%d" % i}
        for i in range(3)
    ]
    ctx = WinContext(signals=sig, render_state={"units_summary": both})
    assert evaluate(spec, ctx) is True
    assert sig.then_progress["handoff"] == 2


# ── C. SOLVENCY / NO-CHEAT: scripted policies on the live engine ─────

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level  # noqa: E402


def _split_by_type(rs):
    us = rs.get("units_summary", []) or []
    jeeps = [str(u["id"]) for u in us if u.get("type") == "jeep"]
    tanks = [str(u["id"]) for u in us if u.get("type") == "2tnk"]
    return jeeps, tanks


def _stall(rs, C):
    return [C.observe()]


def _b_first(rs, C):
    """Send ONLY squad B (tanks) toward its waypoint, ignore A.
    The `then:` chain stays at index 0 forever — clause 0 (jeeps at
    P1) never latches, so clause 1 is never credited."""
    _jeeps, tanks = _split_by_type(rs)
    if not tanks:
        return [C.observe()]
    return [C.move_units(tanks, 90, 30)]


def _single_squad_tour(rs, C):
    """Squad A (jeeps) tours P1 then P2. Clause 1 latches (jeeps at
    P1). Clause 2 demands TANKS at P2 — jeeps don't count — so the
    chain never completes regardless of where the jeeps drive."""
    jeeps, _tanks = _split_by_type(rs)
    if not jeeps:
        return [C.observe()]
    # Just keep sending jeeps to BOTH waypoints — engine moves them.
    return [C.move_units(jeeps, 90, 30)]


def _intended_handoff(rs, C):
    """Parallel dispatch: jeeps to P1, tanks to P2 — B holds at P2
    while A latches at P1, then `then:` advances both clauses."""
    jeeps, tanks = _split_by_type(rs)
    cmds = []
    if jeeps:
        cmds.append(C.move_units(jeeps, 50, 10))
    if tanks:
        cmds.append(C.move_units(tanks, 90, 30))
    return cmds or [C.observe()]


def _make_handoff_policy(legs):
    """Build a stateful policy that drives each squad through its
    own list of waypoints in order — once a leg is reached (>=3
    units of that type in radius 8), it sticks (no oscillation back
    to a satisfied earlier leg if a single unit drifts out). `legs`
    is {unit_type: [(x,y), (x,y), ...]}, executed in order."""
    state = {t: 0 for t in legs}

    def policy(rs, C):
        us = rs.get("units_summary", []) or []
        cmds = []
        for utype, waypoints in legs.items():
            ids = [str(u["id"]) for u in us if u.get("type") == utype]
            if not ids:
                continue
            idx = state[utype]
            # Advance through every consecutive waypoint already
            # satisfied (sticky — never regress).
            while idx < len(waypoints):
                wx, wy = waypoints[idx]
                here = sum(
                    1 for u in us
                    if u.get("type") == utype
                    and (u["cell_x"] - wx) ** 2 + (u["cell_y"] - wy) ** 2
                    <= 8 * 8
                )
                if here >= 3 and idx < len(waypoints) - 1:
                    idx += 1
                else:
                    break
            state[utype] = idx
            tx, ty = waypoints[idx]
            cmds.append(C.move_units(ids, tx, ty))
        return cmds or [C.observe()]

    return policy


def _intended_handoff_medium_factory():
    return _make_handoff_policy({
        "jeep": [(50, 10), (60, 20)],
        "2tnk": [(90, 30)],
    })


def _intended_handoff_hard_factory():
    return _make_handoff_policy({
        "jeep": [(50, 10), (60, 20)],
        "2tnk": [(90, 30), (100, 15)],
    })


# Seeds 1..4 = the held-out seed contract from CLAUDE.md.

@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE on timeout, got {res.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_b_first_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _b_first, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: B-first dash must LOSE (then-latch never "
        f"advances past A's clause), got {res.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_single_squad_tour_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _single_squad_tour, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: single-squad tour (jeeps only) must LOSE "
        f"— the tank-only clause cannot be satisfied — got {res.outcome}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_handoff_wins_easy(seed):
    c = compile_level(load_pack(PACK), "easy")
    res = run_level(c, _intended_handoff, seed=seed)
    assert res.outcome == "win", (
        f"easy seed{seed}: parallel A→P1/B→P2 must WIN, got {res.outcome}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_handoff_wins_medium(seed):
    c = compile_level(load_pack(PACK), "medium")
    res = run_level(c, _intended_handoff_medium_factory(), seed=seed)
    assert res.outcome == "win", (
        f"medium seed{seed}: 3-leg handoff must WIN, got {res.outcome}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_handoff_wins_hard(seed):
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _intended_handoff_hard_factory(), seed=seed)
    assert res.outcome == "win", (
        f"hard seed{seed}: 4-leg interleaved handoff must WIN, got "
        f"{res.outcome}"
    )
