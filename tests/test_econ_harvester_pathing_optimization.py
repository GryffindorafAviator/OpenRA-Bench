"""Full contributor-loop validation for econ-harvester-pathing-optimization.

The pack tests harvester SPLIT-routing under heterogeneous round-trip
cost (OR vehicle-routing / SC2 worker-distribution / M/M/c queueing).
TWO harvesters, TWO patches — A near and B far. The optimal policy
sends ONE harv to A and ONE to B; the win predicate encodes the
routing requirement as `units_of_type_in_region_gte` clauses on BOTH
patch regions plus a modest cash bar.

Bar (per CLAUDE.md "no defect, no cheat"):
- stall LOSES every tier (no income, no region cover).
- 2-on-A LOSES every tier — pumps more credits than the bar but never
  enters B's region, so the routing clause fails.
- 2-on-B LOSES every tier — clears B's region but the FAR round-trip
  earns ~2000 cr, well below the cash bar.
- 1A+1B (intended SPLIT) WINS every tier and every hard seed.
- Hard: a memorised single-pair "always send to (16,14)+(80,14)"
  policy LOSES on the spawn-mismatched seeds (the `any_of` disjunct
  for that spawn pair never matches).
- The capability-policy that IDENTIFIES the matched (A,B) pair from
  the harvs' Y row and splits accordingly WINS every hard seed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-harvester-pathing-optimization.yaml"

# Easy / medium geometry.
EASY_A = (16, 18)
EASY_B = (60, 18)
MED_A = (16, 18)
MED_B = (80, 18)

# Hard tier: four patches, NORTH and SOUTH spawn-matched pairs.
HARD_NA = (16, 14)
HARD_NB = (80, 14)
HARD_SA = (16, 28)
HARD_SB = (80, 28)


# ---------------------------------------------------------------- policies


def stall_policy(rs, Command):
    return [Command.observe()]


def _make_alloc(targets):
    """Send harv[i] (in id order) to targets[i] every turn. The
    `harvest` order persists so re-issuing is idempotent; passing
    `None` for a slot leaves that harv idle (used by the stall-ish
    one-harv probes)."""
    def f(rs, Command):
        harvs = sorted(
            (u for u in rs.get("units_summary", []) if u.get("type") == "harv"),
            key=lambda u: u["id"],
        )
        cmds = []
        for h, t in zip(harvs, targets):
            if t is not None:
                cmds.append(Command.harvest([str(h["id"])], *t))
        return cmds or [Command.observe()]
    return f


def _make_smart_hard():
    """Hard-tier intended policy: identify the matched (A,B) pair from
    the harvs' Y row (NORTH base → harvs at y=14..15 → split to
    (16,14)+(80,14); SOUTH base → y=28..29 → split to (16,28)+(80,28))."""
    def f(rs, Command):
        harvs = sorted(
            (u for u in rs.get("units_summary", []) if u.get("type") == "harv"),
            key=lambda u: u["id"],
        )
        if not harvs:
            return [Command.observe()]
        y = harvs[0]["cell_y"]
        if y < 21:
            targets = [HARD_NA, HARD_NB]
        else:
            targets = [HARD_SA, HARD_SB]
        return [
            Command.harvest([str(h["id"])], *t)
            for h, t in zip(harvs, targets)
        ]
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
    assert pack.meta.id == "econ-harvester-pathing-optimization"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    # The task's three named real-world / benchmark anchors.
    assert any("SC2" in a and "worker" in a for a in anchors), anchors
    assert any("OR" in a and "routing" in a for a in anchors), anchors
    assert any("M/M/c" in a for a in anchors), anchors


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks ≤ ceiling AND
    after_ticks ≤ ceiling AND within_ticks == after_ticks (so a
    non-finisher LOSES, not draws)."""
    pack = load_pack(PACK)

    def _find_within(node):
        """Recurse through nested all_of/any_of to find the
        within_ticks leaf in the win condition."""
        if isinstance(node, dict):
            if "within_ticks" in node:
                return int(node["within_ticks"])
            for k in ("all_of", "any_of"):
                if k in node:
                    for c in node[k]:
                        v = _find_within(c)
                        if v is not None:
                            return v
        return None

    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        wt = _find_within(L.win_condition.model_dump())
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt is not None, f"{lvl}: within_ticks missing from win"
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt == ft, (
            f"{lvl}: within_ticks {wt} != after_ticks {ft} "
            "(non-finisher must LOSE, not draw)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier must define ≥2 agent spawn_point groups (the
    UPGRADED contract — a single memorised opening can't generalise)."""
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
    assert res.outcome == "loss", (
        f"stall must LOSE easy; got {res.outcome} ev={_ev(res)}"
    )


def test_easy_both_to_a_loses_despite_high_cash():
    """The crucial discrimination: 2-on-A earns ~18000 cr (well over
    the 4000 bar) but never enters B's region — the routing clause
    fails so the win predicate as a whole fails. LOSS, not WIN."""
    _, res = _run("easy", lambda: _make_alloc([EASY_A, EASY_A]))
    assert res.outcome == "loss", (
        f"2-on-A must LOSE easy (no harv in B region); "
        f"got {res.outcome} ev={_ev(res)}"
    )
    assert _ev(res) >= 8000, (
        f"2-on-A should still HAVE earned a lot of cash "
        f"(routing clause is the teeth, not cash); ev={_ev(res)}"
    )


def test_easy_both_to_b_loses_on_cash():
    """2-on-B clears B's region but the FAR round-trip earns ~2000 cr
    over 4500 ticks — well below the 4000 bar. LOSS."""
    _, res = _run("easy", lambda: _make_alloc([EASY_B, EASY_B]))
    assert res.outcome == "loss", (
        f"2-on-B must LOSE easy (~2000 ev < 4000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_easy_split_wins():
    """The intended split (1 harv to A, 1 harv to B) clears both
    routing clauses AND the modest cash bar — WIN."""
    _, res = _run("easy", lambda: _make_alloc([EASY_A, EASY_B]))
    assert res.outcome == "win", (
        f"1A+1B split must WIN easy; got {res.outcome} ev={_ev(res)}"
    )


def test_easy_split_wins_either_assignment():
    """The assignment of which harv-id goes where doesn't matter —
    routing is symmetric. Sanity check the reversed assignment also
    wins (catches a hidden id-ordering dependency)."""
    _, res = _run("easy", lambda: _make_alloc([EASY_B, EASY_A]))
    assert res.outcome == "win", (
        f"1B+1A split must WIN easy; got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- MEDIUM


def test_medium_stall_loses():
    _, res = _run("medium", lambda: stall_policy)
    assert res.outcome == "loss", (
        f"stall must LOSE medium; got {res.outcome} ev={_ev(res)}"
    )


def test_medium_both_to_a_loses_despite_high_cash():
    """2-on-A still earns the most credits (~18000) but the B-region
    clause is unsatisfied. LOSS."""
    _, res = _run("medium", lambda: _make_alloc([MED_A, MED_A]))
    assert res.outcome == "loss", (
        f"2-on-A must LOSE medium (no harv in B region); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_medium_both_to_b_loses_on_cash():
    """B at (80,18) is FAR — 2 harvs there earn only ~2000 cr over
    4500 ticks, well below the 5000 bar."""
    _, res = _run("medium", lambda: _make_alloc([MED_B, MED_B]))
    assert res.outcome == "loss", (
        f"2-on-B must LOSE medium (~2000 ev < 5000 bar); "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_medium_split_wins():
    _, res = _run("medium", lambda: _make_alloc([MED_A, MED_B]))
    assert res.outcome == "win", (
        f"1A+1B split must WIN medium; got {res.outcome} ev={_ev(res)}"
    )


def test_medium_split_wins_either_assignment():
    _, res = _run("medium", lambda: _make_alloc([MED_B, MED_A]))
    assert res.outcome == "win", (
        f"1B+1A split must WIN medium; got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- HARD


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_stall_loses_every_seed(seed):
    _, res = _run("hard", lambda: stall_policy, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE hard/seed{seed}; got {res.outcome} ev={_ev(res)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_both_to_matched_a_loses_every_seed(seed):
    """Stacking both harvs on the NORTH-near patch (16,14) is a "use
    only A" policy. On a NORTH spawn it clears the A-region clause
    but the NORTH B-region is empty. On a SOUTH spawn it's outright
    far from both matched regions. LOSS every seed."""
    _, res = _run("hard", lambda: _make_alloc([HARD_NA, HARD_NA]), seed=seed)
    assert res.outcome == "loss", (
        f"2-on-A(north) must LOSE hard/seed{seed}; "
        f"got {res.outcome} ev={_ev(res)}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_both_to_matched_b_loses_every_seed(seed):
    """Mirror: 2-on-B(north) — far patch only, never enters A's
    region (on either spawn) and earns ~1000 cr."""
    _, res = _run("hard", lambda: _make_alloc([HARD_NB, HARD_NB]), seed=seed)
    assert res.outcome == "loss", (
        f"2-on-B(north) must LOSE hard/seed{seed}; "
        f"got {res.outcome} ev={_ev(res)}"
    )


def test_hard_memorised_north_split_loses_on_south_spawn_seeds():
    """A model that memorises "always split (16,14)+(80,14)" loses on
    the SOUTH-spawn seeds (1 and 3 per round-robin) because the harvs
    head to NORTH-matched cells and never enter the SOUTH-matched
    regions — the SOUTH disjunct fails (no harvs in its regions) and
    the NORTH disjunct fails on cash (the harvs are too far from
    their own proc to round-trip efficiently)."""
    for seed in (1, 3):
        _, res = _run("hard", lambda: _make_alloc([HARD_NA, HARD_NB]), seed=seed)
        assert res.outcome == "loss", (
            f"memorised-NORTH-split must LOSE hard/seed{seed} (SOUTH spawn); "
            f"got {res.outcome} ev={_ev(res)}"
        )


def test_hard_memorised_south_split_loses_on_north_spawn_seeds():
    """Symmetric: memorising "always split (16,28)+(80,28)" loses on
    NORTH-spawn seeds 2 and 4."""
    for seed in (2, 4):
        _, res = _run("hard", lambda: _make_alloc([HARD_SA, HARD_SB]), seed=seed)
        assert res.outcome == "loss", (
            f"memorised-SOUTH-split must LOSE hard/seed{seed} (NORTH spawn); "
            f"got {res.outcome} ev={_ev(res)}"
        )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_smart_spawn_matched_split_wins_every_seed(seed):
    """The intended capability — identify the spawn-matched (A,B)
    pair from the agent's harv Y position, then split-route to BOTH
    matched patches — WINS every seed cleanly."""
    _, res = _run("hard", _make_smart_hard, seed=seed)
    assert res.outcome == "win", (
        f"SMART spawn-matched split must WIN hard/seed{seed}; "
        f"got {res.outcome} ev={_ev(res)}"
    )


# ---------------------------------------------------------------- determinism


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and ev."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_alloc([MED_A, MED_B]), seed=2)
    b = run_level(c, _make_alloc([MED_A, MED_B]), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
