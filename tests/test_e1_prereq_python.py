"""Bench-side mirror of `OpenRA-Rust/openra-sim/tests/test_e1_prereq_enforcement.rs`.

Pins that the Rust engine BLOCKS `StartProduction subject=<pid> item=e1`
when the player owns no `barracks`-providing building (no `tent` / no
`barr`). Vendor RA `infantry.yaml` declares
`e1.Buildable.Prerequisites: ~barracks, ~techlevel.infonly`. The `~`
prefix only hides the entry from the build palette UI — it does NOT
make the prerequisite optional.

`has_prerequisites` (world.rs) strips the `~` and matches against the
player's virtual-prerequisite set, which is computed from each owned
building's `ProvidesPrerequisite` traits. A `fact`+`powr`-only base
provides no `barracks`, so every `build('e1')` order issued from the
bench side must produce ZERO `e1` actors. Adding a `tent` flips the
gate (sanity baseline).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR


PACK = PACKS_DIR / "strategy-trilemma.yaml"


def _strip_type(actors, type_name):
    return [
        a for a in actors
        if not (isinstance(a, dict) and a.get("type") == type_name)
    ]


def _compile_medium_without(pack, *types_to_strip):
    """Compile the medium level with the named actor types removed from
    BOTH base.actors AND the level override (the override replaces base
    actors, so stripping base alone leaks the type back in)."""
    modified = pack.model_copy(deep=True)
    actors = modified.base["actors"]
    for t in types_to_strip:
        actors = _strip_type(actors, t)
    modified.base["actors"] = actors

    for level_def in (modified.levels or {}).values():
        overrides = getattr(level_def, "overrides", None) or {}
        if "actors" in overrides:
            override_actors = overrides["actors"]
            for t in types_to_strip:
                override_actors = _strip_type(override_actors, t)
            overrides["actors"] = override_actors
    return modified.compile("medium")


def _spam_build(item):
    def fn(rs, Command):
        return [Command.build(item)]
    return fn


def test_e1_blocked_without_tent_at_bench_level():
    """End-to-end: stripping `tent` from a scenario (and its medium
    override) means the engine's prereq gate fires on every
    `build('e1')` — no e1 actors ever appear, and a brute-army policy
    LOSES the timeout deadline."""
    pack = load_pack(PACK)
    c = _compile_medium_without(pack, "tent")
    res = run_level(c, _spam_build("e1"), seed=1)

    # No e1 actors should appear at any point in the run. signals.units_lost
    # tracks own-units lost, so absence of e1 is best read from outcome +
    # the lack of any units_killed credit (the brute policy would have
    # killed the hunt harasser if e1 ever spawned). A LOSS outcome is
    # the load-bearing assertion; the engine's BLOCKED log line is the
    # ground truth (visible with `pytest -s`).
    assert res.outcome == "loss", (
        f"build('e1') with NO tent must produce zero infantry → no "
        f"ARMY arm satisfied → timeout LOSS; got {res.outcome}"
    )


def test_e1_succeeds_with_tent_at_bench_level():
    """Sanity baseline: the same pack at medium WITH the default tent
    in place — `build('e1')` does produce units (so the LOSS above is
    specifically the prereq gate, not a missing income / cash floor)."""
    pack = load_pack(PACK)
    c = pack.compile("medium")  # unmodified — tent is in the override
    own_unit_counts = []

    def policy(rs, Command):
        units = rs.get("units_summary") or []
        own_unit_counts.append(len(units))
        return [Command.build("e1")]

    run_level(c, policy, seed=1)
    assert max(own_unit_counts) >= 1, (
        "with tent present, build('e1') must produce at least one "
        "e1 actor at some point in the episode (saw max="
        f"{max(own_unit_counts) if own_unit_counts else 0})"
    )
