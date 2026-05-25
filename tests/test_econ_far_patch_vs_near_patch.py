"""econ-far-patch-vs-near-patch — REASONING capability validation.

Capability: under a hard economy deadline, route the harvester to
the NEAR ore patch (short round-trip) rather than the FAR one.
Round-trip latency dominates pile size inside the deadline — the
short-round-trip patch lets the harv complete many delivery cycles,
while the far patch burns the whole episode in transit.

Pack ships its own purpose-built 64x40 arena (replaces the generic
128x40 rush-hour-arena): near patch at x=16, far patch at x=58, a
~12x distance ratio for a sharp throughput gap.

No-cheat bar (CLAUDE.md "no defect, no cheat"):

  Engine footgun closed by this pack: `auto_route_idle_harvesters`
  routes an idle harv to the NEAREST ore. If the harv were pre-
  placed between the patches, stall would auto-route NEAR and WIN
  for free. The harv is therefore pre-placed ADJACENT TO THE FAR
  PATCH; stall then auto-routes FAR (LOSS), and only an explicit
  `Command.harvest(harv, near_x, near_y)` order overrides the
  route — making routing a real decision.

  - stall (observe-only)        → LOSS every tier (harv auto-routes
    FAR → one delivery cycle → bar unmet).
  - route to the FAR patch      → LOSS every tier (same shape as
    stall).
  - intended route NEAR         → WIN every tier.
  - hard tier defines ≥2 agent spawn_point groups (NORTH y=14 /
    SOUTH y=28); a memorised "always route to (16, 14)" opening
    WINS only on NORTH-spawn seeds and LOSES on SOUTH-spawn seeds.

Anchors: OR travel-vs-throughput, SC2 expansion timing, logistics
routing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-far-patch-vs-near-patch.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """Observe-only — harv auto-routes to nearest ore = FAR patch
    (it is pre-placed adjacent to the far patch)."""
    return [Command.observe()]


def _make_route(tx, ty):
    """Issue Command.harvest(harv, tx, ty) ONCE, then observe. The
    harv's user-issued Harvest activity persists until completion."""
    ordered = [False]

    def policy(rs, Command):
        if not ordered[0]:
            harvs = [
                u for u in (rs.get("units_summary") or [])
                if u.get("type") == "harv"
            ]
            if harvs:
                ordered[0] = True
                return [
                    Command.harvest([str(h["id"])], tx, ty) for h in harvs
                ]
        return [Command.observe()]

    return policy


def _make_smart_hard():
    """Hard-tier intended policy: read harv latitude (NORTH y≈14 /
    SOUTH y≈28), then route to the matched near patch (16, 14) or
    (16, 28). WINS every seed."""
    ordered = [False]

    def policy(rs, Command):
        if not ordered[0]:
            harvs = [
                u for u in (rs.get("units_summary") or [])
                if u.get("type") == "harv"
            ]
            if harvs:
                hy = harvs[0]["cell_y"]
                ty = 14 if hy < 20 else 28
                ordered[0] = True
                return [
                    Command.harvest([str(h["id"])], 16, ty) for h in harvs
                ]
        return [Command.observe()]

    return policy


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "far-patch-vs-near-patch arena must load"
    return c, run_level(c, policy, seed=seed)


def _ev(res):
    return res.signals.cash + res.signals.resources


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-far-patch-vs-near-patch"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "travel" in anchors or "logistics" in anchors or "sc2" in anchors


def test_uses_custom_tight_arena():
    """The pack must declare a procedural arena (not a shared rush-hour-
    arena). Width 128 / height 48 keeps the near-vs-far decision crisp
    (FAR patch at x=58 vs NEAR patch at x=24 — ~34 cells apart, enough
    travel cost to dominate the steady-state throughput delta)."""
    pack = load_pack(PACK)
    bm = pack.base_map
    assert isinstance(bm, dict), (
        f"base_map must be a generator spec dict, got {type(bm).__name__}"
    )
    assert bm.get("generator") == "arena"
    assert bm.get("width") == 128
    assert bm.get("height") == 48


def test_harv_pre_placed_adjacent_to_far_patch():
    """The load-bearing anti-cheat invariant: the harv must be
    pre-placed ADJACENT TO THE FAR PATCH so stall auto-routes FAR
    (and LOSES). If the harv were placed between the patches, stall
    would auto-route NEAR and WIN for free, inverting the bar."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # All agent harvs in this pack should sit near x=56 (the FAR
        # patch is at x=58), regardless of spawn_point.
        harvs = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "harv"]
        assert harvs, f"{lvl}: must have at least one pre-placed harv"
        for h in harvs:
            assert h.position[0] >= 50, (
                f"{lvl}: harv at {h.position} must be pre-placed adjacent "
                "to the FAR patch (x≈56) so stall auto-routes FAR; "
                "otherwise stall WINS for free."
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


def test_all_tiers_have_reachable_deadlines():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        # within_ticks may be inside a nested any_of (hard tier).
        wd = L.win_condition.model_dump()

        def _find(node, key):
            if not isinstance(node, dict):
                return None
            for k, v in node.items():
                if k == key:
                    return int(v)
                if k in ("all_of", "any_of"):
                    for child in v:
                        r = _find(child, key)
                        if r is not None:
                            return r
            return None

        wt = _find(wd, "within_ticks")
        ft = _find(L.fail_condition.model_dump(), "after_ticks")
        assert wt is not None, f"{lvl}: must have within_ticks"
        assert ft is not None, f"{lvl}: must have after_ticks"
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert ft == wt + 1, (
            f"{lvl}: after_ticks {ft} must be within_ticks {wt} + 1"
        )


def test_persistent_enemy_sentinel_anti_draw():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        sentinels = [
            a for a in c.scenario.actors
            if a.owner == "enemy"
        ]
        assert sentinels, f"{lvl}: missing enemy sentinel (anti-DRAW)"


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_stall_loses(level):
    """Harv auto-routes FAR → ~1 cycle in 4500 ticks → bar unmet."""
    _, r = _run(level, _stall, seed=1)
    assert r.outcome == "loss", (
        f"{level}: stall must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_stall_loses_every_seed(seed):
    _, r = _run("hard", _stall, seed=seed)
    assert r.outcome == "loss", (
        f"hard seed={seed}: stall must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_far_route_loses(level):
    """Explicit harvest to the FAR patch — same throughput as stall
    (the harv was already there), bar unmet."""
    _, r = _run(level, _make_route(58, 18), seed=1)
    assert r.outcome == "loss", (
        f"{level}: far-route must LOSE; got {r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_intended_near_route_wins(level):
    """Explicit harvest to the NEAR patch — short round-trip cycles
    the harv many times → cash clears the bar."""
    _, r = _run(level, _make_route(16, 18), seed=1)
    assert r.outcome == "win", (
        f"{level}: intended near-route must WIN; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_smart_spawn_matched_wins_every_seed(seed):
    """The intended capability — read harv latitude and route to the
    matched NEAR patch — WINS every seed."""
    _, r = _run("hard", _make_smart_hard(), seed=seed)
    assert r.outcome == "win", (
        f"hard smart seed={seed}: should WIN; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


def test_hard_memorised_north_route_loses_on_south_seeds():
    """A model that memorises "always route to (16, 14)" WINS on
    NORTH-spawn seeds (2, 4) but LOSES on SOUTH-spawn seeds (1, 3) —
    the load-bearing no-memorisation property of the hard tier."""
    for seed in (1, 3):
        _, r = _run("hard", _make_route(16, 14), seed=seed)
        assert r.outcome == "loss", (
            f"hard seed={seed}: memorised-NORTH route must LOSE on "
            f"SOUTH-spawn seed; got {r.outcome} ev={_ev(r)}"
        )


def test_hard_memorised_south_route_loses_on_north_seeds():
    """Symmetric: memorising (16, 28) WINS on SOUTH seeds and LOSES
    on NORTH seeds (2, 4)."""
    for seed in (2, 4):
        _, r = _run("hard", _make_route(16, 28), seed=seed)
        assert r.outcome == "loss", (
            f"hard seed={seed}: memorised-SOUTH route must LOSE on "
            f"NORTH-spawn seed; got {r.outcome} ev={_ev(r)}"
        )


def test_outcomes_are_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_route(16, 18), seed=2)
    b = run_level(c, _make_route(16, 18), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
