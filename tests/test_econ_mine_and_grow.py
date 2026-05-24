"""econ-mine-and-grow — REASONING capability validation.

Capability: place the first Ore Refinery (`proc`) ADJACENT to the
visible ore patch. The agent starts with a Construction Yard, a
Power Plant, and one idle Ore Truck. ONE ore patch is visible on
the ore-cells channel. The decision is GEOGRAPHIC: drop the proc
near the patch (short harvester round-trip → income clears the bar)
or far away at the base (long round-trip → bar missed).

The pack ships its own purpose-built 96x40 arena (`econ-mine-and-
grow-arena`), shrinking the generic 128x40 rush-hour-arena: the
agent's base anchors at x≈10 and the patch at x≈60, so a refinery
placed AT the patch is ~3 cells away whereas one placed AT the base
is ~47 cells away. The far/near throughput gap is the entire test.

No-cheat bar (CLAUDE.md "no defect, no cheat"):

  - stall (only `observe`)          → LOSS (no proc → harv idle, no
    income → after_ticks bites as real LOSS).
  - far placement (proc AT the base) → LOSS (long round-trip per
    bale → cash bar unmet by deadline).
  - intended NEAR placement (proc adjacent to the patch) → WIN every
    level and every hard seed (income clears the bar with margin).
  - hard tier defines ≥2 agent spawn_point groups (NORTH y=12 /
    SOUTH y=28) so a memorised opening that always sites the proc at
    (57, 12) fails on SOUTH seeds.

Anchors: SC2 first-refinery placement, facility-siting near a
raw-resource node, RTS "first refinery on the patch" idiom.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "econ-mine-and-grow.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """No action — harv stays idle (auto-route requires a `proc`)."""
    return [Command.observe()]


def _make_near():
    """Build + place 2nd proc ADJACENT to the ore patch (60, 20).
    `place_building` does not enforce build-adjacency, so dropping
    the proc at (57, 20) is legal. The auto-route hook installs a
    Harvest activity on the idle harv the moment the proc exists."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        if not s["placed"] and s["turn"] >= 18:
            s["placed"] = True
            return [Command.place_building("proc", 57, 20)]
        return [Command.observe()]

    return policy


def _make_far():
    """Build + place proc AT THE BASE (far from the patch). Round-
    trip per bale is ~47 cells each way → throughput too slow → cash
    bar unmet by deadline."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        if not s["placed"] and s["turn"] >= 18:
            s["placed"] = True
            return [Command.place_building("proc", 14, 20)]
        return [Command.observe()]

    return policy


def _make_hard_smart():
    """Hard-tier intended policy: read the harv's latitude (NORTH y≈12
    or SOUTH y≈28), then site the proc adjacent to the patch on the
    MATCHING latitude (57, 12) or (57, 28). Either spawn → WIN."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        if not s["placed"] and s["turn"] >= 18:
            harvs = [
                u for u in (rs.get("units_summary") or [])
                if u.get("type") == "harv"
            ]
            if harvs:
                hy = harvs[0]["cell_y"]
                # closer to NORTH patch (60,12) or SOUTH (60,28)?
                py = 12 if abs(hy - 12) < abs(hy - 28) else 28
                s["placed"] = True
                return [Command.place_building("proc", 57, py)]
        return [Command.observe()]

    return policy


def _make_hard_memorised_north():
    """Memorise NORTH placement (57, 12). LOSES on SOUTH spawn seeds
    because the harv is at y=28 and the proc at (57, 12) is 16 cells
    away vertically — round-trip latency too slow."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        if not s["placed"] and s["turn"] >= 18:
            s["placed"] = True
            return [Command.place_building("proc", 57, 12)]
        return [Command.observe()]

    return policy


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "mine-and-grow arena must be Rust-loadable"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-mine-and-grow"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2" in anchors or "facility" in anchors or "refinery" in anchors


def test_uses_custom_tight_arena():
    """The pack must declare its purpose-built 96x40 arena (not the
    generic 128x40 rush-hour-arena). Width 96 keeps the near/far gap
    decisive — base at x=10, patch at x=60, ~47 cells of delta."""
    pack = load_pack(PACK)
    bm = pack.base_map
    assert isinstance(bm, dict), (
        f"base_map must be a generator spec dict, got {type(bm).__name__}: "
        "the mine-and-grow task needs a tight arena that makes the "
        "near-vs-far throughput gap the only decisive axis."
    )
    assert bm.get("generator") == "arena"
    assert bm.get("name") == "econ-mine-and-grow-arena"
    assert bm.get("width") == 96
    assert bm.get("height") == 40


def test_ore_patches_declared_at_pack_or_level():
    """At least the easy + medium tiers must declare a single ore
    patch on the east side (x=60, y=20) for the near-vs-far decision
    to bite."""
    pack = load_pack(PACK)
    # Pack-level default patches (used by easy/medium).
    pp = pack.ore_patches or []
    if pp:
        assert any(
            int(p.get("x", 0)) == 60 and int(p.get("y", 0)) == 20
            for p in pp
        ), f"expected patch at (60, 20), got {pp}"


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: ≥2 agent spawn_point groups so the engine round-
    robins the base latitude. The ore patches rotate to match
    (NORTH at y=12, SOUTH at y=28); a memorised opening fails."""
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
            "(non-finisher must LOSE one tick past win clause)"
        )


def test_persistent_enemy_sentinel_anti_draw():
    """An unarmed enemy `fact` sentinel keeps the engine's
    ConquestVictoryConditions from auto-`done`-ing on enemy-
    elimination before the win/fail evaluates."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        sentinels = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert sentinels, f"{lvl}: missing enemy `fact` sentinel (anti-DRAW)"


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, building_types=(), tick=1000, cash=0):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=0,
        own_buildings=[(t, 0, 0) for t in building_types],
        own_building_types=set(building_types),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_easy():
    c = compile_level(load_pack(PACK), "easy")
    # has proc + cash bar met + in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(building_types=("fact", "powr", "proc"), tick=2000, cash=3000),
    )
    # missing proc → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(building_types=("fact", "powr"), tick=2000, cash=9999),
    )
    # cash below bar → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(building_types=("fact", "powr", "proc"), tick=2000, cash=2999),
    )
    # deadline passed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(building_types=("fact", "powr", "proc"), tick=3602, cash=0),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_stall_loses(level):
    """No proc placed → harv idle → 0 income → after_ticks LOSS."""
    _, r = _run(level, _stall, seed=1)
    assert r.outcome == "loss", (
        f"{level}: stall must LOSE; got {r.outcome} cash={r.signals.cash}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_stall_loses_every_seed(seed):
    """Stall LOSES on every hard seed (no proc → no income on either
    spawn)."""
    _, r = _run("hard", _stall, seed=seed)
    assert r.outcome == "loss", (
        f"hard seed={seed}: stall must LOSE; got {r.outcome} "
        f"cash={r.signals.cash}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_far_placement_loses(level):
    """Proc placed AT the base (~47 cells from the patch) → long
    round-trip per bale → cash bar unmet by deadline → LOSS."""
    _, r = _run(level, _make_far(), seed=1)
    assert r.outcome == "loss", (
        f"{level}: far placement must LOSE; got {r.outcome} "
        f"cash={r.signals.cash}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_intended_near_placement_wins(level):
    """Proc placed ADJACENT to the patch → short round-trip → cash
    bar cleared comfortably before deadline → WIN."""
    _, r = _run(level, _make_near(), seed=1)
    assert r.outcome == "win", (
        f"{level}: intended near placement should WIN; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_smart_spawn_matched_wins_every_seed(seed):
    """The intended capability — read the harv's latitude and site
    the proc adjacent to the matching patch — WINS every hard seed."""
    _, r = _run("hard", _make_hard_smart(), seed=seed)
    assert r.outcome == "win", (
        f"hard smart seed={seed}: should WIN; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and cash."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_near(), seed=2)
    b = run_level(c, _make_near(), seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash,
    )
