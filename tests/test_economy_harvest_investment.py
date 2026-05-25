"""Full contributor-loop validation for economy-harvest-investment.

The pack tests early-revenue capital reinvestment under an indivisible
budget: starting_cash is EXACTLY one harvester (1100). A committed
single-path reinvestment (DEEP — second harv on the near patch;
GEOGRAPHIC on easy — second harv driven to a far patch) WINS; the
documented decoy WIDE (a second refinery, blocked by engine
prerequisites) LOSES; hedge (a cheap decoy then a late harv) LOSES on
medium/hard; stall and baseline (no reinvest) LOSE every tier.

The bar is binding on real harvest income (post-S0/S1, Task #14), with
a tick-aligned deadline (within_ticks + after_ticks at the same
boundary) so a non-finisher LOSES — never draws.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "economy-harvest-investment.yaml"

NEAR = (22, 18)
FAR_N = (40, 8)
FAR_S = (40, 28)


# ---------------------------------------------------------------- policies


def stall_policy(rs, Command):
    """No reinvestment, no harvest — pure idle. The strict floor."""
    return [Command.observe()]


def baseline_policy(rs, Command):
    """Pre-placed harv only; engine auto-routes idle harvesters to the
    nearest patch via `auto_route_idle_harvesters`. This represents
    the "no reinvestment" floor — earns income via auto-route but
    cannot clear any tier's bar."""
    return [Command.observe()]


def _make_commit_deep():
    """DEEP: build one extra harv. The engine's auto-route puts the
    new harv on the closest patch (NEAR, by path distance) the moment
    it spawns. Issuing explicit `Command.harvest(...)` every turn
    DISRUPTS the auto-route — let the engine handle routing."""
    state = {"queued": False}

    def f(rs, Command):
        cmds = []
        if not state["queued"] and rs.get("cash", 0) >= 1100:
            cmds.append(Command.build("harv"))
            state["queued"] = True
        return cmds or [Command.observe()]

    return f


def _make_commit_wide():
    """WIDE attempt: queue a second `proc`. The engine blocks the
    order ('missing prerequisites for proc'), so income reverts to
    baseline — the documented decoy path that LOSES every tier."""
    state = {"queued": False}

    def f(rs, Command):
        cmds = []
        if not state["queued"]:
            cmds.append(Command.build("proc"))
            state["queued"] = True
        return cmds or [Command.observe()]

    return f


def _make_commit_geo(far_target):
    """GEOGRAPHIC: build one extra harv, drive it to the far patch
    via `move_units` first; the harv's auto-route otherwise picks the
    NEAR patch. Only issue the harvest order once the new harv has
    arrived near the far patch — don't spam `harvest` to the starter
    harv (it disrupts the engine's auto-cycle)."""
    state = {"queued": False, "moved": set()}

    def f(rs, Command):
        cmds = []
        if not state["queued"] and rs.get("cash", 0) >= 1100:
            cmds.append(Command.build("harv"))
            state["queued"] = True
        harvs = sorted(
            (u for u in rs.get("units_summary", []) if u.get("type") == "harv"),
            key=lambda u: u["id"],
        )
        # Leave the first (starter) harv alone — let it auto-route.
        # Steer the second to the far patch then issue harvest once.
        for i, u in enumerate(harvs):
            if i == 0:
                continue
            uid = str(u["id"])
            fx, fy = far_target
            if uid not in state["moved"]:
                cmds.append(
                    Command.move_units([uid], target_x=fx - 2, target_y=fy)
                )
                if abs(u["cell_x"] - fx) <= 5:
                    state["moved"].add(uid)
            else:
                # One-shot harvest order at the far cell — the harv's
                # FSM then cycles patch ↔ proc.
                cmds.append(Command.harvest([uid], fx, fy))
        return cmds or [Command.observe()]

    return f


def _make_hedge():
    """Hedge: queue a decoy `tent` (Allied Barracks, 500cr) AND a
    `silo` (150cr) BEFORE the harv (total 650cr of distractions),
    pushing cash from 1100 to 450. The agent must wait for harvest
    income to refill cash before the second harv can be queued, losing
    several turns of double-income.

    NOTE: hard tier's hedge can occasionally clear the bar by a slim
    margin (engine income variance ±~300 ev per spawn). The medium
    bar bites hedge cleanly on every seed; hard tolerates a marginal
    1-2-seed hedge-WIN as documented engine income noise — the
    strict LOSS bar holds for STALL, WIDE, baseline."""
    state = {"phase": 0}

    def f(rs, Command):
        cmds = []
        if state["phase"] == 0:
            cmds.append(Command.build("tent"))
            cmds.append(Command.build("silo"))
            state["phase"] = 1
        elif state["phase"] == 1 and rs.get("cash", 0) >= 1100:
            cmds.append(Command.build("harv"))
            state["phase"] = 2
        return cmds or [Command.observe()]

    return f


# ---------------------------------------------------------------- helpers


def _run(level, policy_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    return c, run_level(c, policy_factory(), seed=seed)


def _ev(res):
    return res.signals.cash + res.signals.resources


# ---------------------------------------------------------------- structural


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.status == "active", (
        "pack must be un-quarantined post-S0/S1"
    )
    assert pack.meta.id == "economy-harvest-investment"
    assert pack.meta.capability == "reasoning"
    # Multi-anchor benchmark spec (binding via tests/test_benchmark_anchor_required.py)
    anchors = pack.meta.benchmark_anchor
    assert any("PlanBench" in a for a in anchors)
    assert any("SC2LE" in a for a in anchors)
    assert any("lmgame" in a for a in anchors)
    assert any("reinvestment" in a for a in anchors)


def test_all_tiers_have_reachable_deadlines():
    """Every tier's `within_ticks` and the `after_ticks` fail clause
    must sit AT-OR-BELOW the engine ceiling 93 + 90*(max_turns - 1)
    so the clock actually bites (CLAUDE.md). Tick-alignment idiom."""
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
        assert wt == ft, (
            f"{lvl}: within_ticks {wt} != after_ticks {ft} "
            "(non-finisher must LOSE, not draw)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """The hard tier upgrades the spawn-variation contract (≥2
    `spawn_point` groups); seeds round-robin the start position."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"


def test_hard_spawn_groups_have_starter_harv_without_cell_collisions():
    """Each hard spawn needs the advertised starter harv, and it must not
    share a cell with another agent actor."""
    c = compile_level(load_pack(PACK), "hard")
    by_spawn = {}
    for a in c.scenario.actors:
        if a.owner != "agent":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        by_spawn.setdefault(sp, []).append(a)

    assert len(by_spawn) >= 2
    for sp, actors in by_spawn.items():
        harvs = [a for a in actors if a.type == "harv"]
        assert len(harvs) == 1, f"spawn {sp}: expected one starter harv, got {harvs}"

        occupied = {}
        for a in actors:
            pos = tuple(a.position)
            assert pos not in occupied, (
                f"spawn {sp}: {a.type} overlaps {occupied[pos]} at {pos}"
            )
            occupied[pos] = a.type


# ---------------------------------------------------------------- intended WINS


def test_commit_deep_wins_easy():
    _, res = _run("easy", _make_commit_deep)
    assert res.outcome == "win", f"DEEP must win easy; got {res.outcome} ev={_ev(res)}"


def test_commit_deep_wins_medium():
    _, res = _run("medium", _make_commit_deep)
    assert res.outcome == "win", f"DEEP must win medium; got {res.outcome} ev={_ev(res)}"


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_commit_deep_wins_hard_every_seed(seed):
    _, res = _run("hard", _make_commit_deep, seed=seed)
    assert res.outcome == "win", (
        f"DEEP must win hard/seed{seed}; got {res.outcome} ev={_ev(res)}"
    )


def test_commit_wide_decoy_loses_every_tier():
    """The literal WIDE path (2nd refinery) is engine-blocked; a
    plausible RTS commitment to it must LOSE — the no-defect bar."""
    for lvl in ("easy", "medium", "hard"):
        _, res = _run(lvl, _make_commit_wide)
        assert res.outcome == "loss", (
            f"WIDE decoy must LOSE {lvl}; got {res.outcome} ev={_ev(res)}"
        )


def test_commit_geographic_loses_easy_post_auto_route():
    """GEOGRAPHIC (drive 2nd harv to a far patch via move_units) is
    now a DOCUMENTED DECOY path (post-auto-route engine update). The
    move_units detour costs ~26 cells of transit and the far-patch
    cycle time is much longer than the NEAR cycle; net yield ~11000
    ev — BELOW even the stall floor (~11100) since the harv spent
    cash to build but produced less than the pre-placed harv would
    have. The strict bar: GEOGRAPHIC LOSES, DEEP WINS. (Originally
    documented as a winning path on the loose bar but the engine's
    auto-route ruleset has made the DEEP path uniformly better.)"""
    for far in (FAR_N, FAR_S):
        _, res = _run("easy", lambda f=far: _make_commit_geo(f))
        assert res.outcome == "loss", (
            f"GEOGRAPHIC (far={far}) is now a decoy path; expected LOSS, "
            f"got {res.outcome} ev={_ev(res)}"
        )


# ---------------------------------------------------------------- hedge LOSES


def test_hedge_loses_medium():
    """Spending on the cheap decoy FIRST and only then funding the
    harv costs ~800 ev of lost double-income — the medium bar bites."""
    _, res = _run("medium", _make_hedge)
    assert res.outcome == "loss", (
        f"hedge must LOSE medium; got {res.outcome} ev={_ev(res)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hedge_loses_hard_every_seed(seed):
    """The hard bar separates clean DEEP from delayed hedge.

    With the starter harv no longer overlapping a power plant, DEEP
    reaches at least 21500 EV on every seed, while the cheap-decoy hedge
    tops out at 20850 EV. The 21000 hard bar keeps the reinvestment
    decision load-bearing on both spawn latitudes."""
    _, res = _run("hard", _make_hedge, seed=seed)
    assert res.outcome == "loss", (
        f"hard seed{seed}: hedge must LOSE; got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- floor LOSES


def test_stall_loses_every_tier_as_reachable_loss():
    """A staller must LOSE — never draw. The fail_condition's
    after_ticks must trigger before max_turns ends."""
    for lvl in ("easy", "medium", "hard"):
        c, res = _run(lvl, lambda: stall_policy)
        assert res.outcome == "loss", (
            f"stall must LOSE {lvl} as reachable timeout; got "
            f"{res.outcome} (tick={res.signals.game_tick}, "
            f"max_ticks_in_pack=93+90*({c.max_turns}-1))"
        )


def test_baseline_loses_every_tier():
    """Harvest-only-with-the-starting-harv (no reinvestment) clears
    the income floor but cannot meet any tier's bar."""
    for lvl in ("easy", "medium", "hard"):
        _, res = _run(lvl, lambda: baseline_policy)
        assert res.outcome == "loss", (
            f"baseline must LOSE {lvl}; got {res.outcome} ev={_ev(res)}"
        )


# ---------------------------------------------------------------- determinism


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome and ev.
    The whole contributor loop's reproducibility hinges on this."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_commit_deep(), seed=2)
    b = run_level(c, _make_commit_deep(), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
