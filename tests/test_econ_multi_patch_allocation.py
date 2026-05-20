"""Full contributor-loop validation for econ-multi-patch-allocation.

The pack tests Weber-multi-source / SC2-mineral-patch allocation: 3 (or
4 at hard) ore patches at varied distances from a single refinery, the
agent owns 3 harvesters, and yield-per-harvester scales inversely with
round-trip travel. The capability under test is "distance dominates
throughput; prioritise the NEAR patch" — NOT naive "one harv per
source" diversification.

Bar (per CLAUDE.md "no defect, no cheat"):
- stall LOSES every tier.
- All-to-FAR LOSES every tier (~1000 cr/harv/4500t).
- All-to-MID LOSES medium/hard (~2000 cr/harv).
- Uniform 1-per-patch LOSES medium (~11000 cr < 14000 bar).
- Wrong-NEAR (memorised cell that matches one spawn but not the other)
  LOSES on hard's mismatched seeds.
- Intended capability — 2+ harvs on the spawn-matched NEAR patch — WINS
  every tier and every seed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-multi-patch-allocation.yaml"

# Base/medium patch positions
NEAR = (16, 18)
MID = (40, 18)
FAR = (80, 18)

# Hard tier: 4 patches; NEAREST flips per seed.
P_NORTH = (16, 14)   # NEAREST for spawn_point 0 (NORTH base)
P_SOUTH = (16, 28)   # NEAREST for spawn_point 1 (SOUTH base)
H_MID = (40, 18)
H_FAR = (80, 18)


# ---------------------------------------------------------------- policies


def stall_policy(rs, Command):
    return [Command.observe()]


def _make_alloc(targets):
    """Send harv[i] (in id order) to targets[i] every turn. The
    `harvest` order persists so re-issuing is idempotent."""
    def f(rs, Command):
        harvs = sorted(
            (u for u in rs.get("units_summary", []) if u.get("type") == "harv"),
            key=lambda u: u["id"],
        )
        cmds = [Command.harvest([str(h["id"])], *t) for h, t in zip(harvs, targets)]
        return cmds or [Command.observe()]
    return f


def _make_smart_hard():
    """Hard-tier intended policy: identify the matched NEAR patch from
    the harvs' Y row (NORTH base → harvs at y=13..15 → near is (16,14);
    SOUTH base → y=27..29 → near is (16,28)), then allocate all 3."""
    def f(rs, Command):
        harvs = sorted(
            (u for u in rs.get("units_summary", []) if u.get("type") == "harv"),
            key=lambda u: u["id"],
        )
        if not harvs:
            return [Command.observe()]
        y = harvs[0]["cell_y"]
        target = P_NORTH if y < 20 else P_SOUTH
        return [Command.harvest([str(h["id"])], *target) for h in harvs]
    return f


# ---------------------------------------------------------------- helpers


def _run(level, policy_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    policy = policy_factory() if callable(policy_factory) else policy_factory
    return c, run_level(c, policy, seed=seed)


def _ev(res):
    return res.signals.cash + res.signals.resources


# ---------------------------------------------------------------- structural


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.status == "active"
    assert pack.meta.id == "econ-multi-patch-allocation"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert any("SC2LE" in a for a in anchors)
    assert any("Weber" in a for a in anchors)
    assert any("supply-chain" in a for a in anchors)
    assert any("queueing" in a for a in anchors)


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks ≤ ceiling AND
    after_ticks ≤ ceiling AND within_ticks == after_ticks (so a
    non-finisher LOSES, not draws)."""
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
    """Hard tier must define ≥2 spawn_point groups so different seeds
    place the agent at different starts (the capability test: identify
    the NEAREST patch from YOUR base, don't memorise a fixed cell)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


# ---------------------------------------------------------------- EASY


def test_easy_stall_loses():
    _, res = _run("easy", lambda: stall_policy)
    assert res.outcome == "loss", f"stall must LOSE easy; got {res.outcome} ev={_ev(res)}"


def test_easy_all_to_far_loses():
    _, res = _run("easy", lambda: _make_alloc([FAR, FAR]))
    assert res.outcome == "loss", (
        f"all-to-FAR must LOSE easy (~2000 ev < 8000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_easy_both_to_near_wins():
    _, res = _run("easy", lambda: _make_alloc([NEAR, NEAR]))
    assert res.outcome == "win", (
        f"both-to-NEAR must WIN easy; got {res.outcome} ev={_ev(res)}"
    )


def test_easy_split_near_far_wins():
    """Even the inefficient split (1 NEAR + 1 FAR) clears 8000 because
    the NEAR harv alone supplies ~8000 cr; this is the loose-bar easy
    tier — any allocation that USES the NEAR patch passes."""
    _, res = _run("easy", lambda: _make_alloc([NEAR, FAR]))
    assert res.outcome == "win", (
        f"split-NEAR+FAR must WIN easy; got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- MEDIUM


def test_medium_stall_loses():
    _, res = _run("medium", lambda: stall_policy)
    assert res.outcome == "loss", (
        f"stall must LOSE medium; got {res.outcome} ev={_ev(res)}"
    )


def test_medium_all_to_far_loses():
    _, res = _run("medium", lambda: _make_alloc([FAR, FAR, FAR]))
    assert res.outcome == "loss", (
        f"all-to-FAR must LOSE medium (~3000 ev < 14000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_medium_all_to_mid_loses():
    _, res = _run("medium", lambda: _make_alloc([MID, MID, MID]))
    assert res.outcome == "loss", (
        f"all-to-MID must LOSE medium (~6000 ev < 14000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_medium_uniform_split_loses():
    """The NAIVE one-harv-per-patch heuristic LOSES medium — the
    capability test is "transport cost dominates, not parallelism"."""
    _, res = _run("medium", lambda: _make_alloc([NEAR, MID, FAR]))
    assert res.outcome == "loss", (
        f"uniform 1/1/1 split must LOSE medium (~11000 ev < 14000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_medium_one_near_two_mid_loses():
    """A "diversify slightly towards MID" allocation still under-uses
    the NEAR patch; medium's bar bites at this margin."""
    _, res = _run("medium", lambda: _make_alloc([NEAR, MID, MID]))
    assert res.outcome == "loss", (
        f"1-NEAR+2-MID must LOSE medium (~12000 ev < 14000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_medium_balanced_2near_1mid_wins():
    """The intended balanced allocation (2 harvs on NEAR + 1 on MID)
    wins cleanly — the textbook Weber-multi answer with this geometry."""
    _, res = _run("medium", lambda: _make_alloc([NEAR, NEAR, MID]))
    assert res.outcome == "win", (
        f"2-NEAR+1-MID (intended) must WIN medium; got {res.outcome} "
        f"ev={_ev(res)}"
    )


def test_medium_2near_1far_wins():
    """Variant balanced allocation also clears the bar (the NEAR
    saturation is soft enough that an extra FAR harv adds ~1000 ev)."""
    _, res = _run("medium", lambda: _make_alloc([NEAR, NEAR, FAR]))
    assert res.outcome == "win", (
        f"2-NEAR+1-FAR must WIN medium; got {res.outcome} ev={_ev(res)}"
    )


def test_medium_all_to_near_wins():
    """Concentrating ALL harvs on the NEAR patch is also a valid
    optimum at this fleet size (3 harvs don't saturate the patch hard);
    the bar discriminates "ignored NEAR" from "used NEAR", not from
    "balanced vs concentrated"."""
    _, res = _run("medium", lambda: _make_alloc([NEAR, NEAR, NEAR]))
    assert res.outcome == "win", (
        f"all-to-NEAR must WIN medium; got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- HARD


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_stall_loses_every_seed(seed):
    _, res = _run("hard", lambda: stall_policy, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE hard/seed{seed}; got {res.outcome} ev={_ev(res)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_all_to_far_loses_every_seed(seed):
    _, res = _run("hard", lambda: _make_alloc([H_FAR, H_FAR, H_FAR]), seed=seed)
    assert res.outcome == "loss", (
        f"all-to-FAR must LOSE hard/seed{seed} (~4500 ev < 22000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_uniform_1pn_1mid_1far_loses_every_seed(seed):
    """The uniform "one per source" heuristic that drops one of the
    two near patches loses every seed — too much load on transport-
    expensive patches."""
    _, res = _run("hard", lambda: _make_alloc([P_NORTH, H_MID, H_FAR]), seed=seed)
    assert res.outcome == "loss", (
        f"uniform 1-PN+1-MID+1-FAR must LOSE hard/seed{seed}; "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_hard_memorised_pn_loses_on_south_spawn_seeds():
    """A model that memorises "always send to (16,14)" loses on
    SOUTH-base seeds (1 and 3 per round-robin) — the matched NEAR
    patch is (16,28), and (16,14) is now ~14 cells of vertical
    travel from the proc, dropping yield to ~16500 ev < 22000."""
    for seed in (1, 3):
        _, res = _run("hard", lambda: _make_alloc([P_NORTH, P_NORTH, P_NORTH]), seed=seed)
        assert res.outcome == "loss", (
            f"memorised-PN must LOSE hard/seed{seed} (SOUTH spawn); "
            f"got {res.outcome} ev={_ev(res)}"
        )


def test_hard_memorised_ps_loses_on_north_spawn_seeds():
    """Symmetric: memorising (16,28) loses on NORTH-base seeds 2 and 4."""
    for seed in (2, 4):
        _, res = _run("hard", lambda: _make_alloc([P_SOUTH, P_SOUTH, P_SOUTH]), seed=seed)
        assert res.outcome == "loss", (
            f"memorised-PS must LOSE hard/seed{seed} (NORTH spawn); "
            f"got {res.outcome} ev={_ev(res)}"
        )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_smart_spawn_matched_wins_every_seed(seed):
    """The intended capability — identify the spawn-matched NEAR patch
    from the agent's own base position, then concentrate harvs there —
    WINS every seed cleanly."""
    _, res = _run("hard", _make_smart_hard, seed=seed)
    assert res.outcome == "win", (
        f"SMART spawn-matched policy must WIN hard/seed{seed}; "
        f"got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- determinism


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and ev."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_alloc([NEAR, NEAR, MID]), seed=2)
    b = run_level(c, _make_alloc([NEAR, NEAR, MID]), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
