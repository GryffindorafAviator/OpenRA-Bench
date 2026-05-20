"""tempo-double-window: strict pulsed-load duty cycle (strike-lull-strike).

The advertised capability is a temporally-extended ON-OFF-ON plan:
hit Cluster A in window 1, idle through a lull (off the through-route),
then hit Cluster B in window 2. The win condition must enforce all
three phases so that the four illegal policies (stall, continuous
attack, all-in-window-1, brute-rush-to-B) all LOSE while the intended
strike-lull-strike policy WINS, on every level and every hard seed.

Predicate semantics: `all_of[{units_killed_gte:3}, {units_killed_gte:7}]`
semantically collapses to the higher bar — so the "≥3 kills before the
lull" half of the original predicate sketch cannot be expressed by
stacked kill counts. The strict double-window is therefore enforced by
the stateful `waypoint_sequence` latch over
(Cluster A → lull safe-point → Cluster B): a unit must touch each
region in that exact order before the final win can register. These
tests pin that invariant.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "tempo-double-window.yaml"

# Waypoint anchors (must match the pack).
A_STRIKE = (35, 20)
LULL = (55, 36)
B_STRIKE = (105, 20)


def _sig(tick: int = 0, killed: int = 0, lost: int = 0):
    return types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        seq_progress={},
    )


def _ctx(sig, units_xy):
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


# ── Pack compiles, three levels exist, hard has ≥2 spawn groups ───────


def test_pack_compiles_all_three_levels():
    p = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(p, lvl)
        assert c.map_supported, f"{lvl}: map must be Rust-loadable"
        assert c.fail_condition is not None, f"{lvl}: must have fail_condition"
        # within_ticks reachable inside max_turns (engine ≈90 tk/turn).
        tick_max = 93 + 90 * (c.max_turns - 1)
        assert 4500 <= tick_max, (
            f"{lvl}: within_ticks=4500 not reachable in max_turns={c.max_turns}"
        )
        assert 4501 <= tick_max, (
            f"{lvl}: after_ticks fail=4501 not reachable in max_turns="
            f"{c.max_turns} → would draw, not lose"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """The hard tier must split agent actors across ≥2 `spawn_point`
    groups (so seed round-robin actually varies the start). Pulsed
    tempo is still the SOLE controlled variable; only the staging
    latitude rotates."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_hard_has_two_tsla_emplacements():
    """Hard difficulty axis vs medium: a second `tsla` widens the
    lethal corridor on the through-route, so any attempt to push
    through during the lull burns through the tighter loss cap."""
    c = compile_level(load_pack(PACK), "hard")
    tslas = [a for a in c.scenario.actors if a.type == "tsla"]
    assert len(tslas) >= 2, f"hard must have ≥2 tsla, got {len(tslas)}"


def test_medium_has_one_tsla():
    c = compile_level(load_pack(PACK), "medium")
    tslas = [a for a in c.scenario.actors if a.type == "tsla"]
    assert len(tslas) == 1, f"medium must have exactly 1 tsla, got {len(tslas)}"


def test_easy_has_no_tsla():
    """Easy is the bare pulsed-cycle test: the lull WP enforces the
    duty cycle, no through-route punishment yet."""
    c = compile_level(load_pack(PACK), "easy")
    tslas = [a for a in c.scenario.actors if a.type == "tsla"]
    assert len(tslas) == 0, f"easy must have 0 tsla, got {len(tslas)}"


def test_hard_loss_cap_tighter_than_medium():
    """Difficulty axis: hard tightens `units_lost_lte` from 4 → 2."""
    med = compile_level(load_pack(PACK), "medium")
    hard = compile_level(load_pack(PACK), "hard")

    def _cap(cond):
        extras = dict(getattr(cond, "__pydantic_extra__", {}) or {})
        for c in extras.get("all_of", []):
            ex = dict(c if isinstance(c, dict) else getattr(
                c, "__pydantic_extra__", {}) or {})
            if "units_lost_lte" in ex:
                return int(ex["units_lost_lte"])
        return None

    assert _cap(med.win_condition) == 4
    assert _cap(hard.win_condition) == 2


# ── The intended strike-lull-strike WINS ──────────────────────────────


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_intended_strike_lull_strike_wins(lvl):
    c = compile_level(load_pack(PACK), lvl)
    sig = _sig(tick=600, killed=4, lost=1)
    # Phase 1 — column reaches Cluster A and engages.
    evaluate(c.win_condition, _ctx(sig, [A_STRIKE] * 4))
    assert sig.seq_progress["T"] == 1
    # Lull — column retreats south to the safe-point and waits.
    sig.game_tick = 1500
    evaluate(c.win_condition, _ctx(sig, [LULL] * 4))
    assert sig.seq_progress["T"] == 2
    # Phase 2 — after the lull opens, column pushes to Cluster B and
    # finishes with the required kills and within the loss cap.
    sig.game_tick = 3200
    sig.units_killed = 7
    sig.units_lost = 1 if lvl == "hard" else 3
    assert evaluate(c.win_condition, _ctx(sig, [B_STRIKE] * 4)) is True


# ── Stall LOSES on the clock (every level) ────────────────────────────


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_stall_loses_on_clock(lvl):
    c = compile_level(load_pack(PACK), lvl)
    sig = _sig(tick=4501, killed=0, lost=0)
    # Column never left the staging area — no waypoints reached.
    ctx = _ctx(sig, [(6, 20)] * 6)
    assert evaluate(c.win_condition, ctx) is False
    assert evaluate(c.fail_condition, ctx) is True
    # And `after_ticks: 4501` is reachable inside max_turns.
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


# ── Continuous attack LOSES on medium/hard (tsla bleeds units) ────────


@pytest.mark.parametrize("lvl", ["medium", "hard"])
def test_continuous_attack_loses_on_tsla_attrition(lvl):
    """No retreat to the lull WP, and the column gets shredded crossing
    the Tesla corridor between the clusters."""
    c = compile_level(load_pack(PACK), lvl)
    cap = 4 if lvl == "medium" else 2
    sig = _sig(tick=300, killed=3, lost=0)
    # Cluster A engaged.
    evaluate(c.win_condition, _ctx(sig, [A_STRIKE] * 6))
    assert sig.seq_progress["T"] == 1
    # No lull retreat — push straight through. Tesla bleed → over cap.
    sig.game_tick = 3200
    sig.units_killed = 7
    sig.units_lost = cap + 1
    ctx_b = _ctx(sig, [B_STRIKE] * 2)
    # Even if the agent now reaches B, lull was skipped → latch stuck.
    assert sig.seq_progress["T"] == 1  # never advanced past 1
    # Reaching B without the lull cannot advance the latch (order is
    # latched, monotonic) — the win never becomes satisfiable.
    assert evaluate(c.win_condition, ctx_b) is False
    assert sig.seq_progress["T"] == 1
    # And the loss-cap bust ⇒ fail_condition is true.
    assert evaluate(c.fail_condition, ctx_b) is True


def test_continuous_attack_also_loses_on_easy_via_skipped_lull():
    """Easy has no Tesla, but the ordered latch alone is enough: a
    continuous push that skips the lull WP can NEVER satisfy the win
    (no matter the kill count), so the clock runs out → LOSS."""
    c = compile_level(load_pack(PACK), "easy")
    sig = _sig(tick=300, killed=3, lost=0)
    evaluate(c.win_condition, _ctx(sig, [A_STRIKE] * 6))
    # March straight through to B several times — latch is stuck at 1.
    for tick in (1000, 2000, 3000, 4000, 4500):
        sig.game_tick = tick
        sig.units_killed = 7
        evaluate(c.win_condition, _ctx(sig, [B_STRIKE] * 6))
    assert sig.seq_progress["T"] == 1
    # Clock expires while latch is stuck ⇒ real LOSS.
    sig.game_tick = 4501
    ctx = _ctx(sig, [B_STRIKE] * 6)
    assert evaluate(c.win_condition, ctx) is False
    assert evaluate(c.fail_condition, ctx) is True


# ── All-in-window-1 LOSES: spent the force at A, can't reach B ────────


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_all_in_window_1_loses_insufficient_force_for_window_2(lvl):
    """The column commits everything to A in window 1, then has too
    few survivors / too few kills left for window 2 — the ≥7 kill bar
    isn't met by Cluster A alone (only 4 enemies there)."""
    c = compile_level(load_pack(PACK), lvl)
    sig = _sig(tick=600, killed=4, lost=2)
    evaluate(c.win_condition, _ctx(sig, [A_STRIKE] * 4))
    # Retreat to lull, then push to B — but only 2 units remain and
    # they only manage 1 more kill at B before the clock runs out.
    sig.game_tick = 1500
    evaluate(c.win_condition, _ctx(sig, [LULL] * 2))
    sig.game_tick = 4000
    sig.units_killed = 5      # < 7 → win bar not met
    sig.units_lost = 4 if lvl != "hard" else 2
    ctx = _ctx(sig, [B_STRIKE] * 2)
    assert evaluate(c.win_condition, ctx) is False
    # Now the clock expires ⇒ real LOSS, not a draw.
    sig.game_tick = 4501
    ctx = _ctx(sig, [B_STRIKE] * 2)
    assert evaluate(c.fail_condition, ctx) is True


# ── Timing teeth: the lull is timed (after_ticks gates window 2) ──────


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_reaching_b_too_early_does_not_win(lvl):
    """Even if the ordered chain A→lull→B is finished, the win must
    NOT register before `after_ticks: 3000` — that's the lull window."""
    c = compile_level(load_pack(PACK), lvl)
    sig = _sig(tick=300, killed=7, lost=0)
    evaluate(c.win_condition, _ctx(sig, [A_STRIKE] * 4))
    sig.game_tick = 600
    evaluate(c.win_condition, _ctx(sig, [LULL] * 4))
    sig.game_tick = 1500   # chain finished but lull not timed out
    ctx = _ctx(sig, [B_STRIKE] * 4)
    assert evaluate(c.win_condition, ctx) is False
    # Past the lull threshold → win finally registers.
    sig.game_tick = 3100
    assert evaluate(c.win_condition, _ctx(sig, [B_STRIKE] * 4)) is True


# ── Reachable timeout: within_ticks ≤ tick_max(max_turns) ─────────────


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_timeout_is_reachable_within_max_turns(lvl):
    """No draw degeneracy: the deadline must bite inside max_turns."""
    c = compile_level(load_pack(PACK), lvl)
    tick_max = 93 + 90 * (c.max_turns - 1)
    # within_ticks=4500 — the win deadline.
    assert 4500 <= tick_max
    # The fail deadline (after_ticks: 4501) must also be reachable.
    assert 4501 <= tick_max


# ── Real engine smoke: pack runs on the live Rust env for one episode ─


def test_pack_runs_on_live_engine_smoke():
    """A staller policy completes max_turns and resolves as LOSS on the
    live engine (catches map / actor / wiring panics; no model needed)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), "medium")

    def stall(_rs, Command):
        return [Command.observe()]

    res = run_level(c, stall, seed=1)
    # The staller never reaches A → latch never advances → no win.
    # The clock expires before max_turns ⇒ outcome must be "loss"
    # (not a draw — the whole point of a reachable after_ticks fail).
    assert res.outcome == "loss", (
        f"expected stall to LOSE on medium, got {res.outcome} "
        f"(turns={res.turns}, kills={res.signals.units_killed}, "
        f"losses={res.signals.units_lost})"
    )
