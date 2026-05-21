"""End-to-end guardrail: a BUILT pillbox (`pbox`) is an active
direct-fire tower.

RA's `pbox` is an `AttackGarrisoned` defense — in C# its offensive
power comes from infantry loaded into its `Cargo`, so the YAML carries
NO direct `Armament` trait. The engine does not model garrisoning, so
before the `pbox`-weapon fix the auto-target loop's `weapons.first()`
returned `None` and a built pbox stood completely inert: defense packs
that built pbox towers were not actually testing an active-defense
capability — pre-placed `e1` defenders carried the kills.

`GameRules::from_ruleset` now assigns the canonical RA anti-infantry
pillbox weapon `M60mg` to garrison-only ground-turret defenses (pbox,
hbox) when they carry no explicit `Armament`. This test pins the
bench-side round-trip:

* a pre-placed agent `pbox` with an enemy `e1` in range auto-fires and
  kills the e1 (no orders issued — the engine's defense auto-target
  loop drives it);
* an agent that BUILDS a `pbox` via `build` + `place_building` next to
  an enemy e1 also kills it.

Mirror of the Rust test `OpenRA-Rust/openra-sim/tests/test_pbox_fires.rs`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import run_level
from openra_bench.scenarios.loader import compile_level, load_pack


# A tiny scenario: a pre-placed agent pbox with an enemy e1 standing
# 3 cells away (inside the M60mg 4-cell range). A far-east enemy fact
# is the anti-DRAW marker (keeps the episode alive past the e1 kill so
# the win/fail check still has a frame to run). The agent also has a
# fact so the engine has an agent base.
_PREPLACED_PACK = textwrap.dedent(
    """\
    meta:
      id: pbox-fires-preplaced
      title: 'Pillbox auto-fire smoke (pre-placed)'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'active static defense engages an in-range threat'
    base_map: rush-hour-arena
    starting_cash: 1000
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 4000}
      actors:
        - {type: fact, owner: agent, position: [20, 20]}
        - {type: pbox, owner: agent, position: [40, 20]}
        - {type: e1,   owner: enemy, position: [43, 20], stance: 2}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      easy:
        description: 'pre-placed pbox auto-fires on an in-range enemy e1'
        starting_cash: 1000
        win_condition: {units_killed_gte: 1}
        fail_condition: {after_ticks: 3601}
        max_turns: 40
      medium:
        description: 'pre-placed pbox auto-fires on an in-range enemy e1'
        starting_cash: 1000
        win_condition: {units_killed_gte: 1}
        fail_condition: {after_ticks: 3601}
        max_turns: 40
      hard:
        description: 'pre-placed pbox auto-fires on an in-range enemy e1'
        starting_cash: 1000
        win_condition: {units_killed_gte: 1}
        fail_condition: {after_ticks: 3601}
        max_turns: 40
    """
)


# A build-and-place scenario: the agent starts with a fact + tent (so
# the Defense queue can build pbox) and must build a pbox and place it
# next to an enemy e1 to kill it.
_BUILD_PACK = textwrap.dedent(
    """\
    meta:
      id: pbox-fires-build
      title: 'Pillbox auto-fire smoke (build + place)'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'a built static defense engages an in-range threat'
    base_map: rush-hour-arena
    starting_cash: 2000
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe, build, place_building]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 6000}
      actors:
        - {type: fact, owner: agent, position: [20, 20]}
        - {type: tent, owner: agent, position: [24, 20]}
        - {type: powr, owner: agent, position: [24, 24]}
        - {type: e1,   owner: enemy, position: [43, 20], stance: 2}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      easy:
        description: 'build a pbox, place it next to an enemy e1, kill it'
        starting_cash: 2000
        win_condition: {units_killed_gte: 1}
        fail_condition: {after_ticks: 5401}
        max_turns: 60
      medium:
        description: 'build a pbox, place it next to an enemy e1, kill it'
        starting_cash: 2000
        win_condition: {units_killed_gte: 1}
        fail_condition: {after_ticks: 5401}
        max_turns: 60
      hard:
        description: 'build a pbox, place it next to an enemy e1, kill it'
        starting_cash: 2000
        win_condition: {units_killed_gte: 1}
        fail_condition: {after_ticks: 5401}
        max_turns: 60
    """
)


def _compile(pack_yaml: str):
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="pbox_fires_")
    Path(path).write_text(pack_yaml)
    try:
        return compile_level(load_pack(Path(path)), "easy")
    finally:
        Path(path).unlink(missing_ok=True)


def _stall(rs, C):
    return [C.observe()]


def test_preplaced_pbox_kills_in_range_enemy_e1():
    """A pre-placed pbox with an enemy e1 in range kills it with no
    orders — the engine's defense auto-target loop fires the pbox."""
    c = _compile(_PREPLACED_PACK)
    r = run_level(c, _stall, seed=1)
    assert r.signals.units_killed >= 1, (
        f"a built pbox must auto-fire and kill the in-range enemy e1; "
        f"got units_killed={r.signals.units_killed} outcome={r.outcome}. "
        "(Engine bug: pbox has no Armament ⇒ weapons.first() is None ⇒ "
        "a built pbox never fires.)"
    )
    assert r.outcome == "win", (
        f"the pbox kill must satisfy units_killed_gte:1 ⇒ WIN; "
        f"got {r.outcome} (tick={r.signals.game_tick})"
    )


def test_built_pbox_kills_enemy_e1():
    """An agent that BUILDS a pbox and places it next to an enemy e1
    kills it — the pbox is an active direct-fire tower."""
    c = _compile(_BUILD_PACK)

    def build_pbox(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        if n >= 1:
            return [C.observe()]
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        cmds = []
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        # Place the pbox 3 cells from the enemy e1 at (43,20).
        cmds.append(C.place_building("pbox", 40, 20))
        return cmds

    r = run_level(c, build_pbox, seed=1)
    assert r.signals.units_killed >= 1, (
        f"a freshly built+placed pbox must kill the in-range enemy e1; "
        f"got units_killed={r.signals.units_killed} outcome={r.outcome}"
    )
    assert r.outcome == "win", (
        f"the pbox kill must satisfy units_killed_gte:1 ⇒ WIN; got {r.outcome}"
    )


def test_stall_without_pbox_in_range_does_not_kill():
    """Control: a pure-stall agent that never builds a pbox produces no
    kills — confirms the kill in the build test comes from the pbox, not
    from some ambient engine behaviour."""
    c = _compile(_BUILD_PACK)
    r = run_level(c, _stall, seed=1)
    assert r.signals.units_killed == 0, (
        f"with no pbox built, the agent kills nothing; "
        f"got units_killed={r.signals.units_killed}"
    )
    assert r.outcome == "loss", (
        f"a stall that never builds a pbox times out ⇒ LOSS; got {r.outcome}"
    )
