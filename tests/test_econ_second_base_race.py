"""econ-second-base-race — REASONING capability validation.

Capability: under a hard deadline, drop the 2nd Ore Refinery
(`proc`) INSIDE a specific contested target disk (the second-base
location), rather than at the comfortable home corner. The win
predicate combines a SPATIAL clause (`building_in_region`) with an
economy bar — a home-placed proc clears the bar via the near-base
patch but fails the spatial clause; only a forward placement WINS.

The pack ships its own purpose-built 112x40 arena with a midline
water barrier that splits the map into NW and SE lobes (with a
y=18..23 corridor between them). The agent's base anchors at the NW
corner; the contested patch sits at the SE (88, 30) with a radius-6
disk forgiving placements like (86, 30).

No-cheat bar (CLAUDE.md "no defect, no cheat"):

  - stall (observe-only)        → LOSS (no proc, no income).
  - home placement (proc at NW) → LOSS (cash bar partially met by
    near-base patch BUT spatial clause fails — proc is far from
    the target disk).
  - intended forward placement  → WIN (proc inside the disk, the
    auto-spawn harv lands on the rich SE patch and cycles
    immediately).
  - hard tier defines ≥2 agent spawn_point groups (NW y=8 / SW y=32);
    the contested patch flips to the OPPOSITE-DIAGONAL corner
    (SE (88, 30) for NW spawn / NE (88, 8) for SW spawn). A
    memorised always-(86,30) opening fails on SW seeds.

Anchors: SC2 expansion timing / second-base race, RTS contested-
expand idiom, tempo + spatial placement audit.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-second-base-race.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    return [Command.observe()]


def _make_place(target_xy):
    """Build the 2nd proc and place it at target_xy, retrying every
    turn until the queue completes."""
    s = {"b": False, "p": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        own_procs = sum(
            1 for b in (rs.get("own_buildings") or []) if b.get("type") == "proc"
        )
        if not s["b"]:
            s["b"] = True
            return [Command.build("proc")]
        if not s["p"] and s["turn"] >= 16:
            if own_procs >= 1:
                s["p"] = True
            else:
                return [Command.place_building("proc", *target_xy)]
        return [Command.observe()]

    return policy


def _make_hard_smart():
    """Hard-tier intended policy: read harv latitude (NW spawn y≈8 /
    SW spawn y≈32), then place the proc inside the contested disk on
    the OPPOSITE-DIAGONAL corner (SE (86, 30) for NW; NE (86, 8) for
    SW)."""
    s = {"b": False, "p": False, "turn": 0, "tgt": None}

    def policy(rs, Command):
        s["turn"] += 1
        if s["tgt"] is None:
            harvs = [
                u for u in (rs.get("units_summary") or [])
                if u.get("type") == "harv"
            ]
            if harvs:
                hy = harvs[0]["cell_y"]
                # NW spawn (y=8) → SE target; SW spawn (y=32) → NE target.
                s["tgt"] = (86, 30) if hy < 20 else (86, 8)
        own_procs = sum(
            1 for b in (rs.get("own_buildings") or []) if b.get("type") == "proc"
        )
        if not s["b"]:
            s["b"] = True
            return [Command.build("proc")]
        if not s["p"] and s["turn"] >= 16 and s["tgt"]:
            if own_procs >= 1:
                s["p"] = True
            else:
                return [Command.place_building("proc", *s["tgt"])]
        return [Command.observe()]

    return policy


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "second-base-race arena must be Rust-loadable"
    return c, run_level(c, policy, seed=seed)


def _ev(res):
    return res.signals.cash + res.signals.resources


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-second-base-race"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2" in anchors or "expansion" in anchors or "tempo" in anchors


def test_uses_custom_arena_with_midline_barrier():
    """Every tier must declare its 112x40 arena with the midline
    water barrier. The split lobes are what make the forward
    commitment a real geographic decision (no easy back-and-forth)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ovs = L.overrides or {}
        bm = ovs.get("base_map") if isinstance(ovs, dict) else None
        # The level overrides supply the arena spec.
        assert isinstance(bm, dict), (
            f"{lvl}: overrides.base_map must be a generator spec dict"
        )
        assert bm.get("generator") == "arena"
        assert bm.get("width") == 112
        assert bm.get("height") == 40
        obs = bm.get("obstacles") or []
        assert len(obs) >= 2, f"{lvl}: must paint the midline water walls"


def test_win_predicate_has_spatial_clause():
    """The load-bearing clause: `building_in_region` with a 6-cell
    radius around the contested patch. Without this, a home-placed
    proc would WIN on cash alone (the near-base patch sustains it),
    masking the SPATIAL capability test."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        text = str(L.win_condition.model_dump())
        assert "building_in_region" in text, (
            f"{lvl}: win must include building_in_region (spatial clause)"
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
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch"
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
    """No proc → no income → spatial AND cash clauses both fail."""
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
def test_home_placement_loses(level):
    """Proc placed at home (NW corner) — auto-spawn harv lands on
    the near-base patch and runs an economy, BUT the spatial clause
    requires the proc inside the SE disk. LOSS on spatial clause."""
    _, r = _run(level, _make_place((14, 8)), seed=1)
    assert r.outcome == "loss", (
        f"{level}: home placement must LOSE on spatial clause; got "
        f"{r.outcome} ev={_ev(r)}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_intended_forward_placement_wins(level):
    """Proc placed inside the contested SE disk at (86, 30) — the
    auto-spawn harv lands on the rich patch and cycles immediately,
    clearing both clauses before the deadline."""
    _, r = _run(level, _make_place((86, 30)), seed=1)
    assert r.outcome == "win", (
        f"{level}: intended forward placement must WIN; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_smart_diagonal_wins_every_seed(seed):
    """Intended hard-tier capability — read the agent's spawn
    latitude and place the proc inside the OPPOSITE-DIAGONAL disk —
    WINS every seed (1-4)."""
    _, r = _run("hard", _make_hard_smart(), seed=seed)
    assert r.outcome == "win", (
        f"hard smart seed={seed}: should WIN; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_place((86, 30)), seed=2)
    b = run_level(c, _make_place((86, 30)), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
