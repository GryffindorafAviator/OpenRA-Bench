"""Building & Planning scenario family, full loop on Rust.

No-cheat redesign (closer look): the pack tests construction planning
under an ENGINE-ENFORCED tech-tree dependency (a power-less barracks
never completes) plus spatial placement and base relocation. NO power
is pre-placed, so respecting the prerequisite is the real decision.

These tests prove, with deterministic scripted agents (no model):

* the intended "architect" (power → tech-dependent barracks → placed
  defensive line) WINS every level;
* every lazy/greedy/brute/stall policy LOSES every level (the timeout
  fail_condition makes non-win a real LOSS, not a draw), including the
  greedy "build the goal building first" policy — which proves the
  tech prerequisite is genuinely enforced by the engine.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

PACK = PACKS_DIR / "building-and-planning.yaml"

# Region the defensive line must occupy (None on easy = placement-free).
_REGION = {"easy": None, "medium": (40, 20), "hard": (60, 20)}


def _architect_easy(render_state, Command):
    """Legal build order: power first (engine blocks a power-less
    barracks), then the power-dependent barracks."""
    bt = [b["type"] for b in render_state.get("own_buildings", [])]
    prod = render_state.get("production", []) or []
    if "powr" not in bt:
        cmds = [Command.build("powr")] if "powr" not in prod else []
        return cmds + [Command.place_building("powr", 14, 18)]
    if "tent" not in bt:
        cmds = [Command.build("tent")] if "tent" not in prod else []
        return cmds + [Command.place_building("tent", 18, 18)]
    return [Command.observe()]


def _architect_chain(region):
    """powr→tent→pbox chain, creeping the build radius east until the
    two pillboxes can be placed inside `region`."""
    rx, ry = region

    def fn(render_state, Command):
        bt = [
            (b["type"], b["cell_x"], b["cell_y"])
            for b in render_state.get("own_buildings", [])
        ]
        prod = render_state.get("production", []) or []
        types = [b[0] for b in bt]
        eastmost = max([b[1] for b in bt if b[0] == "powr"], default=10)
        if eastmost < rx - 4:
            cmds = [Command.build("powr")] if "powr" not in prod else []
            return cmds + [
                Command.place_building("powr", min(eastmost + 8, rx - 2), 18)
            ]
        if "tent" not in types:
            cmds = [Command.build("tent")] if "tent" not in prod else []
            return cmds + [Command.place_building("tent", rx - 2, ry + 2)]
        npb = sum(b[0] == "pbox" for b in bt)
        if npb < 2:
            cmds = [Command.build("pbox")] if "pbox" not in prod else []
            return cmds + [Command.place_building("pbox", rx + npb * 2, ry)]
        return [Command.observe()]

    return fn


def _idle(render_state, Command):
    return [Command.observe()]


def _spam_powr(render_state, Command):
    """Greedy: only ever build power plants — never the dependent
    barracks/pillbox the win requires."""
    prod = render_state.get("production", []) or []
    bt = [(b["type"], b["cell_x"]) for b in render_state.get("own_buildings", [])]
    em = max([x for t, x in bt if t == "powr"], default=10)
    cmds = [Command.build("powr")] if "powr" not in prod else []
    return cmds + [Command.place_building("powr", min(em + 6, 80), 18)]


def _goal_first(render_state, Command):
    """Greedy: try to build the goal building (barracks/pillbox) FIRST
    with NO power — the engine blocks it (it never completes)."""
    bt = [b["type"] for b in render_state.get("own_buildings", [])]
    prod = render_state.get("production", []) or []
    if "tent" not in bt and "pbox" not in bt:
        cmds = []
        if "tent" not in prod and "pbox" not in prod:
            cmds.append(Command.build("tent"))
        cmds.append(Command.place_building("tent", 16, 18))
        cmds.append(Command.place_building("pbox", 40, 20))
        return cmds
    return [Command.observe()]


def test_pack_compiles_with_three_levels_and_no_deploy():
    pack = load_pack(PACK)
    assert pack.meta.id == "building-and-planning"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # `deploy` must NOT be advertised: the installed engine wheel does
    # not transform an MCV into a `fact` (1:1 bench/engine parity).
    c = compile_level(pack, "hard")
    assert "deploy" not in (c.scenario.tools or []), c.scenario.tools


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, not a draw: the fail deadline must
    be strictly below the tick the episode can reach at max_turns
    (tick ≈ 93 + 90·(max_turns-1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    inner = fc["not"]
    deadline = int(inner["within_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert deadline < reachable, (
        f"{level}: fail deadline {deadline} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_architect_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    agent = (
        _architect_easy if level == "easy"
        else _architect_chain(_REGION[level])
    )
    for seed in (1, 2, 3, 4):
        r = run_level(c, agent, seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended architect should win, got "
            f"{r.outcome}; buildings={r.signals.own_buildings}"
        )
        # Construction debited the budget; the tech chain materialised.
        assert r.signals.cash < c.starting_cash
        sc = score_episode(c, r)
        assert sc.outcome == "win" and 0.0 <= sc.composite <= 1.0


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize(
    "policy", ["idle", "spam_powr", "goal_first"]
)
def test_lazy_and_greedy_policies_lose_every_level_and_seed(level, policy):
    """No defect, no cheat: idle/stall, power-spam (ignores the
    dependent building), and goal-first (defies the engine-enforced
    prerequisite) must all LOSE on every level and hard seed."""
    c = compile_level(load_pack(PACK), level)
    fn = {"idle": _idle, "spam_powr": _spam_powr, "goal_first": _goal_first}[
        policy
    ]
    for seed in (1, 2, 3, 4):
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy}: must LOSE (real fail, not a "
            f"draw), got {r.outcome}; buildings={r.signals.own_buildings}"
        )


def test_easy_run_is_deterministic():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, _architect_easy, seed=3)
    b = run_level(c, _architect_easy, seed=3)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must be deterministic"
