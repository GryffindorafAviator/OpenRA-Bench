"""perception-count-the-threat — exact-count perception cell.

Asserts the binding contract for the pack:
  * intended just-enough-scout WINS every level on every hard seed,
  * stall LOSES every level (clock),
  * under-scout (one band only, never the other) LOSES medium/hard
    (`enemies_discovered_gte:K` unmet),
  * over-scout (run every unit through every fog patch — mid-map
    sweep that brushes the stance:3 inner defenders) LOSES medium/hard
    (clock or attrition),
  * the timeout is REACHABLE (within_ticks ≤ 93+90·(max_turns-1) — no
    draw degeneracy), and
  * hard ships ≥2 agent spawn_point groups (seed-driven start).

These are scripted policies — no model, no network. The discrimination
is the whole point of the cell (calibrate WHEN to stop counting), so
the test owns it directly. Cf. perception-frontier-reading and
perception-target-vs-fog for the sibling perception predicates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK_PATH = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "perception-count-the-threat.yaml"
)


def _stall(_rs, Command):
    return [Command.observe()]


def _e1_count(rs):
    return sum(
        1
        for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "e1"
    )


def _intended_easy(rs, Command):
    """Easy: scout K=3 in the near-east cluster, then build EXACTLY 3
    e1 in response (the pack win predicate is
    `enemies_discovered_gte:3 AND unit_type_count_eq:e1:3` + hold).
    Drive every jeep to the (40,5) sight-line, then queue e1 builds
    until the visible count equals the read."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    jeeps = [u for u in units if u.get("type") == "jeep"]
    cmds = [
        # Push to (55,8) — within RA jeep vision (~8 cells) of the
        # cluster at (60,10). (40,5) was 20 cells short of vision
        # range under vendor RA stats.
        Command.move_units([str(u["id"])], target_x=55, target_y=8)
        for u in jeeps
    ] or [Command.observe()]
    # Spawn the matching three rifle infantry — `unit_type_count_eq`
    # is on `e1`, not `jeep`. Stop at 3 so the over-commit fail
    # (`unit_type_count_gte:e1:4`) never trips.
    if _e1_count(rs) < 3 and "e1" not in (rs.get("production") or []):
        cmds.append(Command.build("e1"))
    return cmds


def _intended_med_hard(rs, Command):
    """Medium/hard: two clusters — north (60,10) and south (60,30).
    Split jeeps across both bands then build EXACTLY 5 e1 (the pack
    win predicate is `enemies_discovered_gte:5 AND
    unit_type_count_eq:e1:5` + hold)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    jeeps = [u for u in units if u.get("type") == "jeep"]
    # Sort jeeps by y so the first goes north, second south — stable
    # across reshuffles between turns.
    jeeps_sorted = sorted(jeeps, key=lambda u: u["cell_y"])
    if jeeps_sorted:
        cmds.append(
            Command.move_units(
                [str(jeeps_sorted[0]["id"])], target_x=60, target_y=5
            )
        )
    if len(jeeps_sorted) > 1:
        cmds.append(
            Command.move_units(
                [str(jeeps_sorted[1]["id"])], target_x=60, target_y=33
            )
        )
    if _e1_count(rs) < 5 and "e1" not in (rs.get("production") or []):
        cmds.append(Command.build("e1"))
    return cmds if cmds else [Command.observe()]


def _under_scout(rs, Command):
    """One-band scout: commit ENTIRELY to whichever lane the spawn
    starts in. Caps the agent at K=3 (one cluster only) — short of
    K=5 — so the discovery bar is the binding LOSS clause."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    avg_y = sum(u["cell_y"] for u in units) / len(units)
    if avg_y < 20:
        # NW spawn → only the north band
        return [
            Command.move_units([str(u["id"])], target_x=60, target_y=5)
            for u in units
        ]
    # SW spawn → only the south band
    return [
        Command.move_units([str(u["id"])], target_x=100, target_y=33)
        for u in units
    ]


def _over_scout(rs, Command):
    """Run every unit through every fog patch — concretely, push the
    whole force into the mid-map sweep at (50,20) where the stance:3
    inner defender wall sits. The detour never reaches the genuine
    clusters in time AND/OR the missile wall attrites units."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=50, target_y=20)
        for u in units
    ]


# ── solvency: intended wins every level + every hard seed ─────────────────


def _seeds(level: str) -> tuple[int, ...]:
    # hard rotates the agent spawn by `seed % spawn_point_count`, so we
    # cover both groups across seeds 1..4 (the cadence's curation set).
    return (1, 2, 3, 4) if level == "hard" else (1,)


def _intended(level: str):
    return _intended_easy if level == "easy" else _intended_med_hard


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_wins_every_level_every_seed(level):
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, level)
    for seed in _seeds(level):
        res = run_level(c, _intended(level), seed=seed)
        assert res.outcome == "win", (
            f"intended just-enough-scout must WIN {level} seed={seed}; "
            f"got outcome={res.outcome} seen={len(res.signals.enemies_seen_ids)} "
            f"lost={res.signals.units_lost} tick={res.signals.game_tick}"
        )


# ── stability: every non-win pattern is a real reachable LOSS ─────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_every_level(level):
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, level)
    for seed in _seeds(level):
        res = run_level(c, _stall, seed=seed)
        assert res.outcome == "loss", (
            f"stall must LOSE {level} seed={seed} (clock — no enemies "
            f"discovered); got outcome={res.outcome}"
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_under_scout_loses_medium_and_hard(level):
    """One-band scout caps at K=3 on medium/hard (predicate K=5) →
    `enemies_discovered_gte:5` unmet → LOSS. Easy is intentionally
    permissive (K=3 in a single cluster, no force-split required)."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, level)
    for seed in _seeds(level):
        res = run_level(c, _under_scout, seed=seed)
        assert res.outcome == "loss", (
            f"under-scout must LOSE {level} seed={seed} via discovery bar; "
            f"got outcome={res.outcome} seen={len(res.signals.enemies_seen_ids)}"
        )
        assert len(res.signals.enemies_seen_ids) < 5, (
            f"under-scout {level} seed={seed} accidentally saw ≥5 enemies "
            f"({len(res.signals.enemies_seen_ids)}) — the one-band cap "
            "isn't holding."
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_over_scout_loses_medium_and_hard(level):
    """Drive every unit into the mid-map defender wall at (50,20).
    The detour AND/OR attrition busts at least one of:
      * `within_ticks:3600` (clock — the sweep never finishes counting),
      * `units_lost_lte:1` (the missile wall attrites).
    Either way the pack emits a real LOSS, never a draw."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, level)
    for seed in _seeds(level):
        res = run_level(c, _over_scout, seed=seed)
        assert res.outcome == "loss", (
            f"over-scout must LOSE {level} seed={seed}; got "
            f"outcome={res.outcome} seen={len(res.signals.enemies_seen_ids)} "
            f"lost={res.signals.units_lost}"
        )


# ── tick / max_turns alignment (no draw degeneracy) ───────────────────────


def test_timeout_is_reachable_within_max_turns():
    """Every level's `within_ticks` / `after_ticks` deadline must be
    reachable inside `max_turns` — else the timeout fail-clause never
    bites and a stall draws instead of losing. The engine advances
    ~90 ticks per decision turn; max tick ≈ 93+90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        max_reachable = 93 + 90 * (c.max_turns - 1)
        # Inspect the win + fail predicate trees for explicit tick gates.
        for node in (
            c.win_condition.model_dump(),
            (c.fail_condition.model_dump() if c.fail_condition else {}),
        ):
            _assert_ticks_reachable(node, max_reachable, level)


def _assert_ticks_reachable(node, max_reachable, level):
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        if key in ("all_of", "any_of"):
            for sub in value or []:
                _assert_ticks_reachable(sub, max_reachable, level)
        elif key == "not":
            _assert_ticks_reachable(value, max_reachable, level)
        elif key in ("within_ticks", "after_ticks"):
            assert int(value) <= max_reachable, (
                f"{level}: {key}={value} exceeds reachable tick "
                f"{max_reachable} (max_turns={(max_reachable - 93)//90 + 1}) — "
                "deadline never bites ⇒ draw degeneracy."
            )


# ── hard-tier contract: ≥2 spawn_point groups ─────────────────────────────


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard uses Wave-9 ENEMY-side spawn_point variation (NE-heavy vs
    SE-heavy enemy compositions) — agent base stays fixed every seed."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "hard")
    groups = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is not None
    }
    assert groups == {0, 1}, (
        f"hard must define enemy spawn_points {{0,1}} for seed-driven "
        f"NE-heavy / SE-heavy alternation; got {sorted(groups)}"
    )
