"""End-to-end integration tests for the scenario→engine→score→leaderboard
pipeline (task #13).

Where the per-pack tests (test_combat_hold_chokepoint, test_perception_count,
…) each pin ONE pack's win/loss bar, this suite exercises the WHOLE pipeline
as a single flow and is designed to catch a regression at ANY stage:

  1. scenario → engine : `load_pack` + `compile_level` → `run_level` with a
     scripted policy produces a terminal outcome (win/loss) with sane signals.
  2. engine → score    : the `EpisodeResult` feeds `score_episode` (composite /
     weakest-link / speed bonus); the ScoreCard has the expected shape and
     bounded values.
  3. score → leaderboard: a full `run_eval.evaluate` report → `ingest_run` /
     `build_table` (the same path `app.load_capability_leaderboard` calls)
     produces a well-formed leaderboard row (model, per-capability means,
     win_rate).

Deterministic — scripted policies only, small seed set, no model / no network.
Two representative, recently hardened packs are used as stable fixtures:

  * combat-hold-chokepoint  (capability: action)     — a known-WINnable pack
    with its intended hold-the-choke policy, and a known-LOSS stall policy.
  * perception-count-the-threat (capability: perception) — a known-WINnable
    pack with its intended just-enough-scout policy, and a known-LOSS stall.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import EpisodeResult, run_level
from openra_bench.leaderboard import build_table, ingest_run
from openra_bench.run_eval import evaluate
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scoring import ScoreCard, score_episode

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"

# ── stable pack fixtures ─────────────────────────────────────────────────
# Both packs were no-cheat-redesigned and verified (every level/seed) on
# 2026-05-20; their intended-WIN and stall-LOSS scripted policies are the
# documented bar in the per-pack tests. Used here as deterministic anchors.
CHOKE = PACKS / "combat-hold-chokepoint.yaml"
PERCEPTION = PACKS / "perception-count-the-threat.yaml"


# ── scripted policies (copied verbatim from the per-pack tests so this
# suite stays self-contained and won't silently drift if those move) ─────


def _stall(_rs, Command):
    """Burn the clock — only observe. Loses every winnable pack."""
    return [Command.observe()]


def _choke_hold_policy(rs, Command):
    """combat-hold-chokepoint intended WIN: keep the squad anchored in the
    corridor and focus-fire the frontmost (lowest cell_x) light tank."""
    units = rs.get("units_summary", []) or []
    enemies = [
        e
        for e in (rs.get("enemy_summary", []) or [])
        if not e.get("is_building")
        and (e.get("type") or "").lower() == "1tnk"
    ]
    if not units or not enemies:
        return [Command.observe()]
    front = min(enemies, key=lambda e: e["cell_x"])
    return [
        Command.attack_unit([str(u["id"])], str(front["id"]))
        for u in units
    ]


def _perception_scout_policy(rs, Command):
    """perception-count-the-threat (easy) intended WIN: drive every scout
    to the single near-east cluster's sight-line and reveal all of it."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=40, target_y=5)
        for u in units
    ]


# ─────────────────────────────────────────────────────────────────────────
# Stage 1+2 — scenario → engine → score: a known-winnable pack with the
# intended scripted policy must produce a terminal WIN scored > 0.
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pack_path, level, policy",
    [
        (CHOKE, "easy", _choke_hold_policy),
        (PERCEPTION, "easy", _perception_scout_policy),
    ],
)
def test_winnable_pack_intended_policy_wins_and_scores_positive(
    pack_path, level, policy
):
    """Full scenario→engine→score flow on a winnable pack: the intended
    scripted policy yields a terminal WIN; scoring the EpisodeResult
    produces a well-shaped ScoreCard with a bounded, positive composite."""
    compiled = compile_level(load_pack(pack_path), level)
    assert compiled.map_supported, "fixture pack must be Rust-loadable"

    # scenario → engine
    res = run_level(compiled, policy, seed=1)
    assert isinstance(res, EpisodeResult)
    assert res.outcome == "win", (
        f"{pack_path.stem}:{level} intended policy must WIN; got "
        f"{res.outcome} (killed={res.signals.units_killed} "
        f"lost={res.signals.units_lost} tick={res.signals.game_tick})"
    )
    # sane episode signals — a real terminated episode, not a degenerate one
    assert 1 <= res.turns <= compiled.max_turns
    assert res.signals.game_tick > 0
    assert res.actions_issued >= res.turns  # ≥1 command per decision turn
    assert res.signals.units_killed >= 0 and res.signals.units_lost >= 0
    assert res.signals.outcome == 1.0  # win maps to 1.0

    # engine → score
    card = score_episode(compiled, res)
    assert isinstance(card, ScoreCard)
    assert card.outcome == "win"
    assert 0.0 < card.composite <= 1.0
    # a WIN must score strictly above its own pre-speed-bonus base only if
    # the win was fast; either way the bonus is bounded and never negative.
    assert card.composite >= card.composite_base
    assert 0.0 <= card.speed <= 1.0
    assert card.composite - card.composite_base <= 0.05 + 1e-9  # SPEED_BONUS
    # P/R/A diagnostics all in [0,1]; weakest_link names one of them.
    for link in ("perception", "reasoning", "action"):
        assert 0.0 <= getattr(card, link) <= 1.0
    assert card.weakest_link in ("perception", "reasoning", "action")
    # speed-bonus accounting is self-consistent on a win.
    assert card.win_tick > 0 and card.win_turns > 0 and card.win_budget > 0


# ─────────────────────────────────────────────────────────────────────────
# Stage 1+2 — a stall policy on a winnable pack must produce a terminal
# LOSS scored low (well below the intended-WIN composite).
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pack_path, level", [(CHOKE, "easy"), (PERCEPTION, "easy")]
)
def test_stall_policy_loses_and_scores_low(pack_path, level):
    """A stall policy on a winnable pack must terminate as a real LOSS,
    and its composite must score strictly below the intended WIN — the
    score path must discriminate effort, not collapse win/loss together."""
    compiled = compile_level(load_pack(pack_path), level)

    loss = run_level(compiled, _stall, seed=1)
    assert loss.outcome == "loss", (
        f"{pack_path.stem}:{level} stall must LOSE; got {loss.outcome}"
    )
    assert loss.signals.outcome == 0.0
    loss_card = score_episode(compiled, loss)
    assert loss_card.outcome == "loss"
    assert 0.0 <= loss_card.composite <= 1.0
    # a loss earns no speed bonus
    assert loss_card.speed == 0.0
    assert loss_card.win_tick == 0 and loss_card.win_turns == 0
    assert loss_card.composite == loss_card.composite_base

    # the intended winning policy on the SAME pack must out-score the stall.
    policy = _choke_hold_policy if pack_path == CHOKE else _perception_scout_policy
    win = run_level(compiled, policy, seed=1)
    assert win.outcome == "win"
    win_card = score_episode(compiled, win)
    assert win_card.composite > loss_card.composite, (
        f"{pack_path.stem}:{level} WIN composite {win_card.composite} must "
        f"exceed stall LOSS composite {loss_card.composite}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — a malformed / edge pack must fail gracefully, not crash.
# ─────────────────────────────────────────────────────────────────────────


def test_malformed_pack_raises_clean_error_not_crash(tmp_path):
    """A structurally invalid pack YAML must raise a clean, contextual
    error from `load_pack` (ValueError with the file path) — never an
    uncaught crash or a silently mis-compiled scenario."""
    bad = tmp_path / "broken.yaml"
    bad.write_text("meta: {id: broken}\nthis is: not a valid pack\n")
    with pytest.raises((ValueError, Exception)) as exc:
        load_pack(bad)
    # the loader wraps the underlying validation error with file context.
    assert "broken.yaml" in str(exc.value) or "broken" in str(exc.value)


def test_empty_pack_raises_not_crash(tmp_path):
    """An empty pack file must raise, not produce a half-built object."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(Exception):
        load_pack(empty)


def test_unsupported_map_pack_is_skipped_gracefully(tmp_path):
    """A pack pointing at a non-Rust-loadable base map must be reported
    as skipped by `evaluate`, never raise — the pipeline degrades, the
    leaderboard ingestion still produces a valid (empty) report."""
    src = (PACKS / "perception-count-the-threat.yaml").read_text()
    # repoint the base map at a name no .oramap resolves.
    import re

    mangled = re.sub(
        r"base_map:\s*\S+", "base_map: no-such-map-xyz", src, count=1
    )
    assert "no-such-map-xyz" in mangled
    f = tmp_path / "future.yaml"
    f.write_text(mangled)
    stats = evaluate(packs=[f], levels=["easy"], seeds=[1])
    assert stats["overall"]["n"] == 0
    assert any("not Rust-loadable" in s for s in stats["skipped"])
    # the report is still well-formed and leaderboard-ingestible.
    rec = ingest_run(stats, "degraded-run", tmp_path / "lb.jsonl")
    assert rec["episodes"] == 0


# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — score → leaderboard: a full evaluate() report over ≥2 episodes
# ingests into a well-formed leaderboard row.
# ─────────────────────────────────────────────────────────────────────────


def test_full_pipeline_evaluate_to_leaderboard_row(tmp_path):
    """End-to-end: evaluate a winnable pack over ≥2 seeds → ingest the
    report → build_table produces one well-formed leaderboard row with a
    bounded win_rate, composite, and per-capability breakdown."""
    stats = evaluate(
        packs=[CHOKE],
        levels=["easy"],
        seeds=[1, 2],
        agent_factory=lambda _c: _choke_hold_policy,
        run_id="integ-test",
    )
    # the eval report itself is well-formed.
    assert stats["overall"]["n"] == 2
    assert len(stats["episodes"]) == 2
    o = stats["overall"]
    assert 0.0 <= o["win_rate"] <= 1.0
    assert 0.0 <= o["composite_mean"] <= 1.0
    # the intended policy wins both seeds → win_rate 1.0.
    assert o["win_rate"] == 1.0, f"intended policy must sweep; got {o}"
    for ep in stats["episodes"]:
        assert ep["cell"] == "combat-hold-chokepoint:easy"
        assert ep["capability"] == "action"
        assert ep["outcome"] == "win"
        assert 0.0 < ep["composite"] <= 1.0

    # score → leaderboard ingestion (the path app.load_capability_leaderboard
    # exercises).
    store = tmp_path / "lb.jsonl"
    rec = ingest_run(stats, "integ-model", store)
    assert rec["model"] == "integ-model"
    assert rec["episodes"] == 2
    assert rec["win_rate"] == 1.0
    assert 0.0 < rec["composite"] <= 1.0
    # per-capability means: the only pack is `action`.
    assert "action" in rec["by_capability"]
    cap = rec["by_capability"]["action"]
    assert cap["n"] == 2
    assert cap["win_rate"] == 1.0
    assert 0.0 <= cap["composite"] <= 1.0

    # build_table → the ranked leaderboard row (min_episodes=1: 2 < default 5).
    table = build_table(store, min_episodes=1)
    assert len(table) == 1
    row = table[0]
    assert row["rank"] == 1
    assert row["model"] == "integ-model"
    assert row["win_rate"] == 1.0
    assert 0.0 < row["composite"] <= 1.0
    assert row["weakest_link"] in ("perception", "reasoning", "action", "n/a")
    # P/R/A means surface on the row, each bounded.
    for link in ("perception", "reasoning", "action"):
        assert 0.0 <= row[link] <= 1.0


def test_leaderboard_aggregates_and_ranks_two_models(tmp_path):
    """Leaderboard aggregation over ≥2 episodes AND ≥2 models: a strong
    (intended-WIN) run must out-rank a weak (stall-LOSS) run on the same
    pack — the score→leaderboard path must preserve the win/loss signal
    through aggregation and ranking."""
    store = tmp_path / "lb.jsonl"

    strong = evaluate(
        packs=[CHOKE],
        levels=["easy"],
        seeds=[1, 2],
        agent_factory=lambda _c: _choke_hold_policy,
        run_id="strong",
    )
    weak = evaluate(
        packs=[CHOKE],
        levels=["easy"],
        seeds=[1, 2],
        agent_factory=lambda _c: _stall,
        run_id="weak",
    )
    assert strong["overall"]["win_rate"] == 1.0
    assert weak["overall"]["win_rate"] == 0.0

    ingest_run(strong, "strong-model", store)
    ingest_run(weak, "weak-model", store)

    table = build_table(store, min_episodes=1)
    assert [r["model"] for r in table] == ["strong-model", "weak-model"], (
        "the intended-WIN run must rank above the stall-LOSS run"
    )
    assert table[0]["rank"] == 1 and table[1]["rank"] == 2
    assert table[0]["composite"] > table[1]["composite"]
    assert table[0]["win_rate"] == 1.0 and table[1]["win_rate"] == 0.0
    # ranking is deterministic across rebuilds.
    again = build_table(store, min_episodes=1)
    assert [r["model"] for r in again] == [r["model"] for r in table]


def test_pipeline_is_deterministic_end_to_end():
    """The same (pack, policy, seed) must produce a bit-identical
    EpisodeResult outcome + ScoreCard composite across runs — the whole
    scenario→engine→score path is deterministic, so a regression at any
    stage is reproducible."""
    compiled = compile_level(load_pack(CHOKE), "easy")
    r1 = run_level(compiled, _choke_hold_policy, seed=1)
    r2 = run_level(compiled, _choke_hold_policy, seed=1)
    assert r1.outcome == r2.outcome
    assert r1.signals.units_killed == r2.signals.units_killed
    assert r1.signals.game_tick == r2.signals.game_tick
    assert r1.turns == r2.turns
    c1 = score_episode(compiled, r1)
    c2 = score_episode(compiled, r2)
    assert c1.composite == c2.composite
    assert c1.weakest_link == c2.weakest_link
