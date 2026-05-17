"""Task #13 robustness: the scenario→engine→score→leaderboard pipeline
must fail safe on bad input and be deterministic on good input.

Each test pins an invariant a contributor (or a flaky model) could
otherwise break silently.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR

_HAS_RUST = True
try:  # engine-dependent tests skip cleanly without the wheel
    import openra_train  # noqa: F401
except Exception:  # noqa: BLE001
    _HAS_RUST = False

_META = (
    "meta: {id: x, title: T, capability: perception, "
    "real_world_meaning: aaaaaaaaaaaaaaaaaaaaaa, "
    "robotics_analogue: bbbbbbbbbbbb}\n"
)
_BASE = (
    "base: {agent: {faction: allies}, enemy: {faction: soviet}, "
    "actors: [{type: jeep, owner: agent, position: [5,5]}]}\n"
)


def _pack(text: str) -> Path:
    p = Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text(text)
    return p


# ── Malformed scenarios are rejected, never silently mis-run ──────────
def test_missing_levels_rejected():
    with pytest.raises(ValueError):
        load_pack(_pack(_META + _BASE + "levels: {easy: {description: only here now, "
                  "win_condition: {explored_pct_gte: 1}}}\n"))


def test_unknown_win_predicate_rejected():
    y = (
        _META + _BASE + "levels:\n"
        "  easy: {description: desc easy here, win_condition: {nope_predicate: 1}}\n"
        "  medium: {description: desc med here, win_condition: {explored_pct_gte: 1}}\n"
        "  hard: {description: desc hard here, win_condition: {explored_pct_gte: 1}}\n"
    )
    with pytest.raises(ValueError):
        load_pack(_pack(y))


def test_missing_required_meaning_rejected():
    # real_world_meaning too short → schema rejects (keeps the library
    # meaningful, per CONTRIBUTING).
    bad_meta = (
        "meta: {id: x, title: T, capability: perception, "
        "real_world_meaning: short, robotics_analogue: bbbbbbbbbbbb}\n"
    )
    y = (
        bad_meta + _BASE + "levels:\n"
        "  easy: {description: desc easy here, win_condition: {explored_pct_gte: 1}}\n"
        "  medium: {description: desc med here, win_condition: {explored_pct_gte: 1}}\n"
        "  hard: {description: desc hard here, win_condition: {explored_pct_gte: 1}}\n"
    )
    with pytest.raises(ValueError):
        load_pack(_pack(y))


def test_bundled_packs_all_valid():
    """Every shipped pack must load + compile all three levels — the
    contributor-facing contract and a regression gate for the library."""
    from openra_bench.scenarios.loader import compile_level

    packs = list(PACKS_DIR.glob("*.yaml"))
    assert packs, "no bundled packs found"
    for f in packs:
        if f.name.startswith(("_", "TEMPLATE")):
            continue
        pk = load_pack(f)
        for lvl in ("easy", "medium", "hard"):
            compile_level(pk, lvl)  # raises on any broken level


# ── Leaderboard store is corruption-tolerant ─────────────────────────
def test_leaderboard_tolerates_corrupt_store(tmp_path):
    from openra_bench.leaderboard import build_table, ingest_run

    s = tmp_path / "lb.jsonl"
    ingest_run(
        {"overall": {"n": 10, "win_rate": 0.5, "composite_mean": 0.6},
         "episodes": [], "summary": {}},
        "m",
        s,
    )
    # Append a torn/garbage line (partial write / manual edit).
    with open(s, "a") as f:
        f.write('{"model": "broken", incomplete\n')
    table = build_table(s)  # must not raise
    assert [r["model"] for r in table] == ["m"]


# ── Determinism on good input ────────────────────────────────────────
@pytest.mark.skipif(not _HAS_RUST, reason="Rust env wheel not installed")
def test_run_eval_whole_result_deterministic():
    from openra_bench.run_eval import evaluate

    pk = [PACKS_DIR / "perception-frontier-reading.yaml"]
    a = evaluate(pk, ["easy"], [1, 2])
    b = evaluate(pk, ["easy"], [1, 2])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── Engine tolerates nonsense commands from a model ──────────────────
@pytest.mark.skipif(not _HAS_RUST, reason="Rust env wheel not installed")
def test_bad_model_commands_do_not_crash_or_win():
    """A model emitting invalid build/place/move never crashes the run
    and cannot accidentally satisfy the objective."""
    from openra_bench.eval_core import run_level
    from openra_bench.scenarios.loader import compile_level

    c = compile_level(load_pack(PACKS_DIR / "economy-force-buildup.yaml"), "easy")

    def chaos(render_state, Command):
        return [
            Command.build("totally-not-a-unit"),
            Command.place_building("nope", 99999, 99999),
            Command.move_units(["88888"], -5, -5),
            Command.attack_unit(["1"], "77777"),
        ]

    res = run_level(c, chaos, seed=1)
    assert res.outcome in {"draw", "loss"}, "garbage play must not win"
    assert res.turns >= 1 and len(res.trace) == res.turns
    assert res.signals.game_tick > 0


@pytest.mark.skipif(not _HAS_RUST, reason="Rust env wheel not installed")
def test_custom_map_no_enemy_scenario_runs_from_yaml():
    """Engine fix: a no-enemy scenario on a custom map must NOT instantly
    terminate (enemy-elimination is not a victory condition when the
    scenario placed no enemy). Everything is read from the pack YAML."""
    from openra_bench.eval_core import run_level
    from openra_bench.scenarios.loader import compile_level, resolve_map_path

    pk = load_pack(PACKS_DIR / "custom-map-no-enemy.yaml")
    c = compile_level(pk, "easy")
    # Custom terrain actually resolved (not the rush-hour fallback).
    mp = resolve_map_path(c.scenario.base_map)
    assert mp is not None and mp.name == "singles-maginot.oramap"

    # Idle agent: with no enemy the run must survive past tick 0 (the
    # bug) and only end on win_condition / max_turns.
    idle = run_level(c, lambda rs, C: [C.observe()], seed=1)
    assert idle.signals.game_tick > 50, "no-enemy scenario terminated instantly"
    assert idle.turns >= 1

    # A scout that drives to the YAML-declared region wins via the
    # declarative win_condition alone (no combat involved).
    def scout(rs, C):
        ids = [str(u["id"]) for u in rs.get("units_summary", [])]
        return [C.move_units(ids, 30, 16)] if ids else [C.observe()]

    won = run_level(c, scout, seed=1)
    assert won.outcome == "win", f"YAML reach_region should win, got {won.outcome}"
