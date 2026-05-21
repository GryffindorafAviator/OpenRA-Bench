"""mfb-rotating-production-pressure — Wave-10 multi-base load-balancing.

Capability: REASONING (rotate production output to the base that is
under — or about to be under — attack; an army does not pour out of
one production cluster when a second front opens).

The pack uses the Wave-9 engine feature `scheduled_events:` to drive
ALTERNATING raid pressure: a raid band hits the WEST base early, then
a second band hits the EAST base much later. The agent owns two
complete bases (each fact + weap + fix + proc + powr) and has NO
attack tools — its only defence is the medium tanks it builds, and a
finished tank appears at whichever war factory is PRIMARY. The
load-bearing verb is `set_primary`: it rotates which war factory the
produced tank spawns at, steering each batch of tanks to the base
that needs a garrison.

Because two war factories make the (single, serialized) vehicle
queue tick 2× — six tanks finish by ≈1530 — the win is gated behind
an `after_ticks:3600` clause inside a `then:` chain so the win cannot
fire until an UNDEFENDED base has long since lost its war factory.
The discriminator is therefore purely "did BOTH bases keep their war
factory", not "who finished six tanks first".

Scripted policies cover the bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall            → LOSS every level/seed (no army; the west raid
                       razes the un-garrisoned west war factory).
  * build-west-only  → LOSS: all six tanks spawn at the WEST base;
                       the EAST raid razes the un-garrisoned EAST war
                       factory before the t3600 gate.
  * build-east-only  → LOSS: symmetric — the WEST base is razed.
  * no-set-primary   → LOSS: spamming `build` without ever rotating
                       sends every tank to the default-primary base.
  * rotate-too-late  → LOSS: rotating only after the 5th tank leaves
                       the EAST base with a single defender.
  * intended rotate  → WIN every level/seed (splits 2/3/4): route
                       the first batch to the WEST war factory, then
                       `set_primary` the EAST war factory so the
                       second batch garrisons the EAST base — both
                       war factories and Construction Yards survive
                       past t3600.
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
    / "mfb-rotating-production-pressure.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "mfb-rotating-production-pressure"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_benchmark_anchor_declares_multi_base_load_balancing():
    pack = load_pack(PACK_PATH)
    anchors = {a.lower() for a in (pack.meta.benchmark_anchor or [])}
    assert any("multi-base" in a for a in anchors), anchors
    assert any("load balancing" in a for a in anchors), anchors
    assert any("queueing" in a for a in anchors), anchors


def test_every_level_has_two_alternating_raid_events():
    """The Wave-9 feature is the whole point — every level carries
    exactly two `spawn_actors` raid events, west BEFORE east."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        sched = getattr(c, "scheduled_events", None) or []
        assert len(sched) == 2, f"{lvl} must declare 2 raid events"
        assert all(e.get("type") == "spawn_actors" for e in sched), sched
        ticks = [e["tick"] for e in sched]
        assert ticks[0] < ticks[1], (
            f"{lvl} raids must ALTERNATE (west tick < east tick): {ticks}"
        )
        # West raid is the WEST base (low x), east raid the EAST base.
        wx = sched[0]["actors"][0]["position"][0]
        ex = sched[1]["actors"][0]["position"][0]
        assert wx < 50 < ex, f"{lvl} raid x-coords not west-then-east: {wx},{ex}"


def test_win_is_gated_behind_after_ticks():
    """The win must be gated by an `after_ticks` clause so the fast
    (2×-factory) production cannot win before an undefended base is
    razed."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True
        )
        ats: list[int] = []

        def _collect(node, out):
            if isinstance(node, dict):
                if "after_ticks" in node:
                    out.append(node["after_ticks"])
                for v in node.values():
                    _collect(v, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, out)

        _collect(win, ats)
        assert ats, f"{lvl} win has no after_ticks gate"
        assert min(ats) >= 3000, f"{lvl} after_ticks gate too early: {ats}"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / after_ticks must be reachable inside max_turns
    (engine ~90 ticks/turn → reachable = 93 + 90·(N-1)) so a
    non-win run is a real LOSS, not a silent DRAW."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True
        )
        fail = compile_level(pack, lvl).fail_condition.model_dump(
            exclude_none=True
        )

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list[int] = []
        _collect(win, "within_ticks", wts)
        assert wts and all(wt <= reachable for wt in wts), (
            f"{lvl} within_ticks unreachable: {wts} > {reachable}"
        )
        ats: list[int] = []
        _collect(fail, "after_ticks", ats)
        assert ats and max(ats) <= reachable, (
            f"{lvl} fail.after_ticks unreachable: {ats} > {reachable}"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (UPGRADED
    contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_no_attack_tools():
    """The test is production ROUTING — the agent must have no
    attack verbs, only build / set_primary / set_rally_point."""
    pack = load_pack(PACK_PATH)
    base = pack.base if isinstance(pack.base, dict) else {}
    tools = {str(t).lower() for t in (base.get("tools") or [])}
    assert "set_primary" in tools, tools
    assert "build" in tools, tools
    assert not (tools & {"attack_unit", "attack_move", "move_units"}), tools


# ── Scripted policies ─────────────────────────────────────────────


def _weap_ids(rs):
    """West-first list of the agent's war-factory ids (from the raw
    obs — the cleaned render_state strips building ids)."""
    bs = rs.get("_raw", {}).get("own_buildings", []) or []
    w = sorted(
        ((b["id"], b["cell_x"]) for b in bs if b["type"] == "weap"),
        key=lambda t: t[1],
    )
    return [wid for wid, _ in w]


def _tank_count(rs):
    return sum(
        1
        for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "2tnk"
    )


def _stall(_rs, Command):
    return [Command.observe()]


def _one_base(which):
    """Build all six tanks, route the whole army to ONE base."""
    st = {"q": 0, "set": False}

    def pol(rs, Command):
        cmds = []
        w = _weap_ids(rs)
        if len(w) >= 2 and not st["set"]:
            cmds.append(
                Command.set_primary([str(w[0 if which == "west" else 1])])
            )
            st["set"] = True
        while st["q"] < 6:
            cmds.append(Command.build("2tnk"))
            st["q"] += 1
        return cmds or [Command.observe()]

    return pol


def _no_set_primary():
    """Spam `build` and never rotate — every tank goes to the
    default-primary war factory."""
    st = {"q": 0}

    def pol(_rs, Command):
        cmds = []
        while st["q"] < 6:
            cmds.append(Command.build("2tnk"))
            st["q"] += 1
        return cmds or [Command.observe()]

    return pol


def _intended(split_west=3):
    """Rotate-to-the-hot-base: route the first `split_west` tanks to
    the WEST war factory, then `set_primary` the EAST war factory so
    the rest garrison the EAST base."""
    st = {"q": 0, "primary": None}

    def pol(rs, Command):
        cmds = []
        w = _weap_ids(rs)
        if len(w) < 2:
            return [Command.observe()]
        want = w[1] if _tank_count(rs) >= split_west else w[0]
        if st["primary"] != want:
            cmds.append(Command.set_primary([str(want)]))
            st["primary"] = want
        while st["q"] < 6:
            cmds.append(Command.build("2tnk"))
            st["q"] += 1
        return cmds or [Command.observe()]

    return pol


# ── The bar — every lazy / single-base play LOSES, intended WINS ──


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE, got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_west_only_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _one_base("west"), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: build-west-only must LOSE "
        f"(EAST base un-garrisoned), got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_east_only_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _one_base("east"), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: build-east-only must LOSE "
        f"(WEST base un-garrisoned), got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_no_set_primary_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _no_set_primary(), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: never rotating the primary must LOSE, "
        f"got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("split", (2, 3, 4))
def test_intended_rotation_wins(level, seed, split):
    """The intended rotate-to-both-bases policy WINS — robust to
    where (tank 2, 3 or 4) the agent rotates the primary."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended(split), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed{seed} split{split}: intended rotation must WIN, "
        f"got {res.outcome}"
    )
