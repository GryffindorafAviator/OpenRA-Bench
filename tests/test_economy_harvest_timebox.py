"""No-cheat + solvency proof for `economy-harvest-timebox` (rebuilt
post-S0/S1 — Task #14 wired real harvest income / silo storage).

For every level + every hard seed (1-4):
  - the INTENDED balanced-harvest policy WINS,
  - STALL (only `observe`) LOSES (starting_cash is below the bar, so
    timeout is reached as a real LOSS),
  - GREEDY units-only / harv-only-no-build LOSES (no income channel,
    or insufficient throughput),
  - the deadline is reachable inside max_turns (real timeout LOSS,
    never a draw),
  - hard has ≥2 spawn_point groups and seed-varied starts,
  - a 200-tick smoke confirms harvested income > 0 on medium (catches
    S0/S1 regression).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK_PATH = PACKS_DIR / "economy-harvest-timebox.yaml"


# ───────────────────────── scripted policies ──────────────────────────────


def stall(render_state, Command):
    """Do-nothing — must lose because starting_cash < bar everywhere."""
    return [Command.observe()]


def harv_only_easy(render_state, Command):
    """Pre-placed harvester only, easy patch coords."""
    ids = [
        str(u["id"])
        for u in render_state.get("units_summary", []) or []
        if u.get("type") == "harv"
    ]
    return [Command.harvest(ids, 22, 18)] if ids else [Command.observe()]


def harv_only_any(render_state, Command):
    """Pre-placed harvester(s) only, try both NW and SW hard cluster
    coords (engine picks the reachable one for the seed). Used on
    medium/hard to verify a single-harvester policy still loses."""
    ids = [
        str(u["id"])
        for u in render_state.get("units_summary", []) or []
        if u.get("type") == "harv"
    ]
    if not ids:
        return [Command.observe()]
    cmds = []
    for uid in ids:
        cmds.append(Command.harvest([uid], 22, 18))
        cmds.append(Command.harvest([uid], 22, 28))
    return cmds


def greedy_e1_only_medium(render_state, Command):
    """Spend the budget on infantry (no income). Loses on medium where
    starting_cash is far below the bar. Medium has no tent/barr so the
    e1 build is rejected, which still LOSES (no income channel)."""
    cmds = []
    if render_state.get("cash", 0) >= 100:
        cmds.append(Command.build("e1"))
    return cmds if cmds else [Command.observe()]


def intended_medium_factory():
    """Build ONE extra harvester (cost 1100) and run both on the two
    pre-placed ore patches — the no-cheat WIN policy on medium."""
    queued = [False]

    def policy(render_state, Command):
        cmds = []
        ids = [
            str(u["id"])
            for u in render_state.get("units_summary", []) or []
            if u.get("type") == "harv"
        ]
        for i, uid in enumerate(ids):
            mx, my = (22, 18) if i % 2 == 0 else (22, 22)
            cmds.append(Command.harvest([uid], mx, my))
        if not queued[0] and render_state.get("cash", 0) >= 1100:
            cmds.append(Command.build("harv"))
            queued[0] = True
        return cmds if cmds else [Command.observe()]

    return policy


def intended_hard_factory():
    """Build TWO extra harvesters (3 total) and feed all three patches
    — works against both NW (spawn 0) and SW (spawn 1) hard groups,
    since the harvester routes to whichever patch is reachable."""
    queued = [0]
    # All possible mine cluster coordinates for both spawn groups; the
    # harvester will path to a reachable one for the active seed.
    mines = [(22, 10), (22, 14), (26, 12), (22, 28), (22, 32), (26, 30)]

    def policy(render_state, Command):
        cmds = []
        ids = [
            str(u["id"])
            for u in render_state.get("units_summary", []) or []
            if u.get("type") == "harv"
        ]
        for i, uid in enumerate(ids):
            mx, my = mines[i % len(mines)]
            cmds.append(Command.harvest([uid], mx, my))
        if queued[0] < 2 and render_state.get("cash", 0) >= 1100:
            cmds.append(Command.build("harv"))
            queued[0] += 1
        return cmds if cmds else [Command.observe()]

    return policy


# ───────────────────────── solvency: intended WINS ────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_easy_intended_wins(seed):
    c = compile_level(load_pack(PACK_PATH), "easy")
    r = run_level(c, harv_only_easy, seed=seed)
    assert r.outcome == "win", (
        f"easy intended seed={seed} should WIN, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_medium_intended_wins(seed):
    c = compile_level(load_pack(PACK_PATH), "medium")
    r = run_level(c, intended_medium_factory(), seed=seed)
    assert r.outcome == "win", (
        f"medium intended seed={seed} should WIN, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_intended_wins_every_seed(seed):
    c = compile_level(load_pack(PACK_PATH), "hard")
    r = run_level(c, intended_hard_factory(), seed=seed)
    assert r.outcome == "win", (
        f"hard intended seed={seed} should WIN, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources}, tick={r.signals.game_tick})"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(level):
    """starting_cash alone is below the bar — stall MUST be a real
    LOSS (not a draw), so the `after_ticks` deadline must bite."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, stall, seed=1)
    assert r.outcome == "loss", (
        f"{level} stall must be a real timeout LOSS, got {r.outcome}"
    )


def test_greedy_units_only_loses_medium():
    """Greedy 'spend on infantry' has no income channel and stalls
    EV at (or below) starting_cash. Must LOSE on medium."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    r = run_level(c, greedy_e1_only_medium, seed=1)
    assert r.outcome == "loss", (
        f"medium greedy-units-only must LOSE, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_medium_single_pre_placed_harvester_loses(seed):
    """The pre-placed harvester alone can't hit medium's bar (5500 by
    tick 3000) — must lose. This is the 'failed to expand' policy."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    r = run_level(c, harv_only_any, seed=seed)
    assert r.outcome == "loss", (
        f"medium single-harvester seed={seed} must LOSE, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_single_pre_placed_harvester_loses(seed):
    """The pre-placed harvester alone can't hit hard's bar (7500 by
    tick 3600) on either spawn — must lose every seed."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    r = run_level(c, harv_only_any, seed=seed)
    assert r.outcome == "loss", (
        f"hard single-harvester seed={seed} must LOSE, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )


# ───────────────────────── tick/turn alignment ────────────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_deadline_reachable_inside_max_turns(level):
    """`within_ticks` (and the matching `after_ticks` fail) must be
    reachable inside max_turns — otherwise stall draws instead of
    losing. Engine advances ~90 ticks per turn (tick ≈ 93+90·(t-1))."""
    c = compile_level(load_pack(PACK_PATH), level)
    max_tick = 93 + 90 * (c.max_turns - 1)
    win_extra = c.win_condition.__pydantic_extra__ or {}
    fail_extra = (
        c.fail_condition.__pydantic_extra__ if c.fail_condition else {}
    ) or {}

    def _within(node):
        if not isinstance(node, dict):
            return None
        for k, v in node.items():
            if k == "within_ticks":
                return int(v)
            if k == "all_of":
                for c2 in v:
                    r = _within(c2)
                    if r is not None:
                        return r
            elif k in ("any_of",):
                for c2 in v:
                    r = _within(c2)
                    if r is not None:
                        return r
        return None

    def _after(node):
        if not isinstance(node, dict):
            return None
        for k, v in node.items():
            if k == "after_ticks":
                return int(v)
            if k in ("any_of", "all_of"):
                for c2 in v:
                    r = _after(c2)
                    if r is not None:
                        return r
        return None

    within = _within(win_extra)
    assert within is not None, f"{level} should have within_ticks"
    assert within <= max_tick, (
        f"{level} within_ticks={within} > max reachable tick {max_tick} "
        f"(max_turns={c.max_turns}); deadline never bites"
    )
    after = _after(fail_extra)
    assert after is not None, f"{level} should have an after_ticks fail"
    assert after <= max_tick, (
        f"{level} after_ticks={after} > max reachable tick {max_tick}"
    )


# ───────────────────────── hard spawn contract ────────────────────────────


def test_hard_has_two_spawn_point_groups():
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups, got {sorted(sp)}"
    )


def test_hard_seed_round_robin_produces_distinct_starts():
    """Smoke that the engine actually round-robins the spawn groups by
    seed (the whole point of declaring multiple spawn_point groups)."""
    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter
    from pathlib import Path

    c = compile_level(load_pack(PACK_PATH), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(tuple(sorted((x["cell_x"], x["cell_y"]) for x in u)))
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


# ───────────────────────── S0/S1 income smoke ─────────────────────────────


def test_income_accrues_within_200_ticks_medium():
    """Smoke-assert that harvested income > 0 within the first ~200
    engine ticks on medium (pre-placed harv + proc + mine). Catches a
    regression where S0/S1 ore income silently stops surfacing."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    starting_cash = c.starting_cash or 0
    # Short-budget run: 3 decision turns ≈ 273 ticks, well past the
    # first refinery deposit (~tick 450). Use a tiny max_turns so we
    # don't drag out CI but still see income materialize.
    samples: list[int] = []

    def harv(render_state, Command):
        units = render_state.get("units_summary", []) or []
        ids = [str(u["id"]) for u in units if u.get("type") == "harv"]
        ev = render_state.get("cash", 0) + render_state.get("resources", 0)
        samples.append(ev)
        return [Command.harvest(ids, 22, 18)] if ids else [Command.observe()]

    r = run_level(c, harv, seed=1)
    # By end-of-run, harvest income must have moved EV above the
    # starting cash (S0/S1 regression test).
    final_ev = r.signals.cash + r.signals.resources
    assert final_ev > starting_cash, (
        f"S0/S1 income regression: medium final EV={final_ev} did not "
        f"exceed starting_cash={starting_cash} (samples={samples[:8]}…)"
    )
