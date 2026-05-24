"""econ-multi-patch-allocation — REASONING capability validation.

Capability: under a hard CAPEX cap (starting_cash 1500 ⇒ exactly ONE
proc up front), choose WHICH ore patches to commit refineries to.
The agent has a Construction Yard + Power Plant (no home refinery)
and looks out at FOUR ore patches scattered around the map, EACH
with a pre-positioned idle Ore Truck waiting beside it. The
`auto_route_idle_harvesters` engine hook lights up every owned harv
the moment a single `proc` exists on the agent's side.

Win predicate: `building_count_gte: {type: proc, n: 2}` AND
`cash_gte: M` AND `within_ticks: 3600`. So:
  - stall                → LOSS (no procs, no income).
  - one-proc opening     → LOSS (proc-count clause fails even if
                                  cash is high).
  - two-proc opening at productive patches → WIN.

No-cheat bar (per CLAUDE.md):
  - stall LOSES every tier and every hard seed.
  - one-proc LOSES every tier (proc-count clause).
  - intended (2 procs placed near productive patches) WINS every
    tier and every hard seed (1-4).
  - hard tier defines ≥2 agent spawn_point groups (NORTH y=12 /
    SOUTH y=28) so a memorised cell-coord opening fails.

Anchors: SC2 macro 2-base vs 3-base, OR resource-allocation under a
CAPEX cap, lmgame-Bench resource-streaming.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-multi-patch-allocation.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """Observe-only — no procs ever built, harvs idle indefinitely."""
    return [Command.observe()]


def _make_one_proc(px=14, py=20):
    """Build exactly ONE proc and place it near the base. All four
    pre-staffed harvs auto-route to the lone proc and deposit, so
    cash climbs — but the win predicate requires ≥2 procs. LOSS on
    proc-count clause even though cash easily clears the bar."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        own_procs = sum(
            1 for b in (rs.get("own_buildings") or []) if b.get("type") == "proc"
        )
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        # Retry placement until accepted (queue completion timing varies).
        if not s["placed"] and s["turn"] >= 16:
            if own_procs >= 1:
                s["placed"] = True
            else:
                return [Command.place_building("proc", px, py)]
        return [Command.observe()]

    return policy


def _make_two_procs(p1, p2):
    """Build 2 procs sequentially: first one cheap from starting cash,
    second one funded by deposits from the first. Both placements
    retry every turn until the queue completes."""
    s = {"b1": False, "p1": False, "b2": False, "p2": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        own_procs = sum(
            1 for b in (rs.get("own_buildings") or []) if b.get("type") == "proc"
        )
        if not s["b1"]:
            s["b1"] = True
            return [Command.build("proc")]
        if not s["p1"] and s["turn"] >= 16:
            if own_procs >= 1:
                s["p1"] = True
            else:
                return [Command.place_building("proc", *p1)]
        if s["p1"] and not s["b2"] and rs.get("cash", 0) >= 1400:
            s["b2"] = True
            return [Command.build("proc")]
        if s["b2"] and not s["p2"]:
            if own_procs >= 2:
                s["p2"] = True
            else:
                return [Command.place_building("proc", *p2)]
        return [Command.observe()]

    return policy


def _make_hard_smart():
    """Hard-tier intended policy: read the agent's base latitude
    from the harv positions, then commit two procs — one near the
    base (covers the A-near and centre patches) and one near the
    matched medium patch on the agent's latitude (B_n for NORTH /
    C_s for SOUTH)."""
    s = {"b1": False, "p1": False, "b2": False, "p2": False, "turn": 0,
         "patches": None}

    def policy(rs, Command):
        s["turn"] += 1
        if s["patches"] is None:
            harvs = [
                u for u in (rs.get("units_summary") or [])
                if u.get("type") == "harv"
            ]
            if harvs:
                avg_y = sum(h["cell_y"] for h in harvs) / len(harvs)
                if avg_y < 20:
                    # NORTH spawn (base y=12, harvs at A_n(20,12),
                    # B_n(40,4), centre(40,20), D far). Place 1st
                    # proc near base + A_n, 2nd near B_n.
                    s["patches"] = [(11, 14), (38, 4)]
                else:
                    # SOUTH spawn (base y=28, harvs at A_s(20,28),
                    # centre(40,20), C_s(40,36), D far). Place 1st
                    # proc near base + A_s, 2nd near C_s.
                    s["patches"] = [(11, 30), (38, 36)]
        own_procs = sum(
            1 for b in (rs.get("own_buildings") or []) if b.get("type") == "proc"
        )
        if not s["b1"]:
            s["b1"] = True
            return [Command.build("proc")]
        if not s["p1"] and s["turn"] >= 16 and s["patches"]:
            if own_procs >= 1:
                s["p1"] = True
            else:
                return [Command.place_building("proc", *s["patches"][0])]
        if s["p1"] and not s["b2"] and rs.get("cash", 0) >= 1400:
            s["b2"] = True
            return [Command.build("proc")]
        if s["b2"] and not s["p2"] and s["patches"]:
            if own_procs >= 2:
                s["p2"] = True
            else:
                return [Command.place_building("proc", *s["patches"][1])]
        return [Command.observe()]

    return policy


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "multi-patch arena must be Rust-loadable"
    policy = policy_factory() if callable(policy_factory) else policy_factory
    return c, run_level(c, policy, seed=seed)


def _ev(res):
    return res.signals.cash + res.signals.resources


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-multi-patch-allocation"
    assert pack.meta.capability == "reasoning"
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    # The pack is anchored to multi-source allocation idioms (OR /
    # SC2 / lmgame).
    assert "sc2" in anchors or "or " in anchors or "resource" in anchors


def test_all_tiers_have_reachable_deadlines():
    """Tick-alignment idiom: within_ticks ≤ ceiling AND after_ticks
    ≤ ceiling AND within_ticks + 1 == after_ticks (non-finisher
    LOSES, not draws). Ceiling = 93 + 90·(max_turns - 1)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        wt = next(
            int(c["within_ticks"])
            for c in L.win_condition.model_dump()["all_of"]
            if "within_ticks" in c
        )
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch "
            "(non-finisher must LOSE one tick past win)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else None)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    sp.discard(None)
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


def test_win_predicate_requires_two_procs():
    """The load-bearing distinction: even a single-proc opening with
    cash above the bar LOSES because the win predicate requires
    `building_count_gte: {type: proc, n: 2}`. This is the multi-
    refinery decision under test."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        clauses = L.win_condition.model_dump()["all_of"]
        proc_n = next(
            c["building_count_gte"]
            for c in clauses
            if "building_count_gte" in c
        )
        assert proc_n.get("type") == "proc" and proc_n.get("n") >= 2, (
            f"{lvl}: win must require ≥2 procs; got {proc_n}"
        )


def test_persistent_enemy_sentinel_anti_draw():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        sentinels = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert sentinels, f"{lvl}: missing enemy `fact` sentinel (anti-DRAW)"


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_stall_loses(level):
    """No procs → harvs all idle → cash stuck at 1500 starting →
    after_ticks LOSS."""
    _, r = _run(level, lambda: _stall, seed=1)
    assert r.outcome == "loss", (
        f"{level}: stall must LOSE; got {r.outcome} cash={r.signals.cash}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_stall_loses_every_seed(seed):
    _, r = _run("hard", lambda: _stall, seed=seed)
    assert r.outcome == "loss", (
        f"hard seed={seed}: stall must LOSE; got {r.outcome} "
        f"cash={r.signals.cash}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_one_proc_loses_on_proc_count_clause(level):
    """A single-proc opening: all four pre-staffed harvs auto-route
    and deposit to the lone proc; cash climbs well above the bar BUT
    the win predicate requires ≥2 procs → LOSS on proc-count clause."""
    _, r = _run(level, _make_one_proc, seed=1)
    assert r.outcome == "loss", (
        f"{level}: one-proc must LOSE on proc-count clause; got "
        f"{r.outcome} cash={r.signals.cash}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_intended_two_procs_wins(level):
    """The intended capability — commit two procs at productive
    patches (here: B (40, 8) + C (40, 32)) — clears both the
    proc-count clause and the cash bar before the deadline."""
    _, r = _run(level, lambda: _make_two_procs((38, 10), (38, 30)), seed=1)
    assert r.outcome == "win", (
        f"{level}: intended 2-procs at B+C must WIN; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_smart_spawn_matched_wins_every_seed(seed):
    """The intended hard-tier capability — read the agent's spawn
    latitude and commit two procs (one near the base + A patch, one
    near the matched medium patch on the agent's latitude) — WINS
    every seed (1-4)."""
    _, r = _run("hard", _make_hard_smart, seed=seed)
    assert r.outcome == "win", (
        f"hard smart seed={seed}: should WIN; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_two_procs((38, 10), (38, 30)), seed=2)
    b = run_level(c, _make_two_procs((38, 10), (38, 30)), seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash,
    )
