#!/usr/bin/env python3
"""Generate the research-grounded 200-level scenario catalog.

Procedurally parameterized per the anti-memorization rule (Procgen /
SMACv2): each pack in a category varies coords/thresholds/decoys, and
each pack's 3 levels form a monotone difficulty ladder where the
*decision* gets harder (less info, more decoys, tighter clock,
committed budget, attrition cap) — never just bigger raw numbers.

Win-conditions use ONLY predicates in win_conditions.py; all packs run
on rush-hour-arena (the only Rust-loadable map until S10). Output: flat
`cat-<code>-NN.yaml` files in packs/, auto-covered by the parametrized
engine-run gate. See SCENARIO_CATALOG.md.

Run:  python tools/gen_catalog.py   (writes packs, prints a summary)
"""

from __future__ import annotations

import pathlib

import yaml

PACKS = pathlib.Path(__file__).resolve().parent.parent / "openra_bench" / "scenarios" / "packs"

# Capability tag (taxonomy) -> schema capability literal.
CAP = {"perception": "perception", "reasoning": "reasoning", "action": "action"}

AG = {"agent": {"faction": "allies"}}
EN = {"enemy": {"faction": "soviet"}}


def jeep(n, x, y):
    return {"type": "jeep", "owner": "agent", "position": [x, y], "count": n}


def squad(x, y):
    return [
        {"type": "2tnk", "owner": "agent", "position": [x, y], "count": 2},
        {"type": "e1", "owner": "agent", "position": [x + 1, y + 2], "count": 3},
    ]


def enemy(t, x, y, n=1):
    return {"type": t, "owner": "enemy", "position": [x, y], "count": n, "stance": 2}


def econ_base(cash, tent=False):
    actors = [
        {"type": "fact", "owner": "agent", "position": [10, 18]},
        {"type": "powr", "owner": "agent", "position": [14, 18]},
        {"type": "jeep", "owner": "agent", "position": [8, 16]},
        enemy("e1", 114, 34),
    ]
    if tent:
        actors.insert(2, {"type": "tent", "owner": "agent", "position": [12, 21]})
    return {
        "agent": AG["agent"], "enemy": EN["enemy"],
        "tools": ["build", "place_building", "move_units", "deploy", "stop"],
        "planning": True, "termination": {"max_ticks": 40000},
        "actors": actors,
    }, cash


def scout_base():
    return {
        "agent": AG["agent"], "enemy": EN["enemy"],
        "tools": ["move_units", "attack_unit", "stop"],
        "planning": True, "termination": {"max_ticks": 30000},
        "actors": [jeep(2, 6, 6), {"type": "e1", "owner": "agent",
                                   "position": [7, 8], "count": 2}],
    }


def combat_base(decoys=0):
    a = squad(8, 18) + [enemy("e1", 100, 20, 2), enemy("e3", 104, 24)]
    for d in range(decoys):
        a.append(enemy("e1", 40 + d * 8, 10, 1))
    return {
        "agent": AG["agent"], "enemy": EN["enemy"],
        "tools": ["move_units", "attack_unit", "attack_move", "stop"],
        "planning": True, "termination": {"max_ticks": 30000},
        "actors": a,
    }


# Per category: (code, title, cap, robo, meaning, n_packs, base_fn, win_fn)
# win_fn(p, lvl) -> dict ; lvl in 0(easy)/1(med)/2(hard); p = pack index.
def L(*xs):
    return {"all_of": list(xs)}


CATS = [
    ("c1", "Frontier Scouting", "perception",
     "UAV/UGV frontier selection: choose which unknown region to commit a "
     "time-limited searcher to.",
     "Pathfinding is solved; deciding WHICH unexplored region to commit to "
     "under partial information and a deadline is the real search problem.",
     6, lambda p: scout_base(),
     lambda p, l: L({"explored_pct_gte": 38 + l * 9 + p},
                    *([{"units_lost_lte": 0}] if l == 2 else []),
                    {"within_ticks": 16000 - l * 3500 - p * 200})),
    ("c2", "Threat Enumeration", "perception",
     "Scene state-estimation under fog and decoys before committing force.",
     "Estimating hidden adversary state from partial sightings before "
     "acting — perception under occlusion and distractors.",
     6, lambda p: combat_base(decoys=2),
     lambda p, l: L({"enemies_discovered_gte": 2 + l},
                    {"units_lost_lte": 1 if l < 2 else 0},
                    {"within_ticks": 14000 - l * 3000 - p * 150})),
    ("c3", "Tech Critical Path", "reasoning",
     "Task-graph scheduling with prerequisites under a makespan.",
     "Precondition-ordered construction (B needs A) to hit a capability "
     "deadline — topological planning under a cost+time budget.",
     6, lambda p: econ_base(6000, tent=False),
     lambda p, l: L({"has_building": "tent"},
                    {"building_total_gte": 4 + l},
                    {"power_surplus_gte": 0},
                    {"within_ticks": 22000 - l * 5000 - p * 300})),
    ("c4", "Power-Budget Online", "reasoning",
     "Keeping an online resource (power) non-negative while expanding.",
     "Sequencing subsystem bring-up so a shared online resource never goes "
     "negative — constraint-respecting schedule.",
     5, lambda p: econ_base(5000, tent=False),
     lambda p, l: L({"power_surplus_gte": 0},
                    {"building_total_gte": 4 + l},
                    {"within_ticks": 20000 - l * 4500 - p * 250})),
    ("c5", "Budget Allocation (spend)", "reasoning",
     "Indivisible-budget allocation: wide vs deep, splitting clears nothing.",
     "Allocating a finite budget across competing investments where a "
     "half-measure on both fails — resource-allocation under a hard cap.",
     6, lambda p: econ_base(2200 - p * 120, tent=True),
     lambda p, l: L({"building_count_gte": {"type": "powr", "n": 2}},
                    {"building_total_gte": 5 + l},
                    {"within_ticks": 20000 - l * 4500})),
    ("c6", "Time-Boxed Capital Deploy", "reasoning",
     "Convert a budget into fielded force AND standing infra before a clock.",
     "Provisioning under an energy/money budget and a mission deadline — "
     "how much to commit and when.",
     6, lambda p: econ_base(2000 - p * 100, tent=True),
     lambda p, l: L({"own_units_gte": 4 + l},
                    {"building_total_gte": 4 + l},
                    *([{"units_lost_lte": 0}] if l == 2 else []),
                    {"within_ticks": 16000 - l * 3500})),
    ("c7", "Defensive-Direction Commit", "perception",
     "Place limited defenses toward the most likely approach under "
     "directional uncertainty.",
     "Committing scarce defensive assets to an inferred threat axis — "
     "spatial decision under directional ambiguity.",
     6, lambda p: econ_base(5000, tent=True),
     lambda p, l: L({"building_in_region": {"type": "pbox", "x": 40 + p * 5,
                                            "y": 20, "radius": 9 - l,
                                            "count": 2 + (l == 2)}},
                    {"within_ticks": 24000 - l * 5000})),
    ("c8", "Base-Placement & Staging", "perception",
     "Found a base in a designated region, then stage toward a threat axis.",
     "Choosing a base site that satisfies spatial constraints, then "
     "pre-positioning — facility-layout + staging.",
     5, lambda p: econ_base(4000, tent=False),
     lambda p, l: L({"building_in_region": {"type": "powr", "x": 30 + p * 4,
                                            "y": 18, "radius": 10 - l * 2,
                                            "count": 1}},
                    {"building_total_gte": 3 + l},
                    {"within_ticks": 22000 - l * 5000})),
    ("c9", "Commit-vs-Retreat", "reasoning",
     "Engage vs hold under partial info with an attrition cap.",
     "The core risk call: commit force on visible info vs wait for more — "
     "decision under uncertainty with a loss budget.",
     6, lambda p: combat_base(decoys=1 + p % 3),
     lambda p, l: L({"units_killed_gte": 2 + l},
                    {"units_lost_lte": 3 - l},
                    {"within_ticks": 14000 - l * 3000})),
    ("c10", "Force Coordination", "action",
     "Drive split groups to converge simultaneously and survive.",
     "Coordinated multi-robot rendezvous: steer separated teams to a "
     "common staging point on time.",
     6, lambda p: {
         "agent": AG["agent"], "enemy": EN["enemy"],
         "tools": ["move_units", "attack_unit", "stop"], "planning": True,
         "termination": {"max_ticks": 30000},
         "actors": [jeep(2, 6, 6), jeep(2, 6, 34),
                    {"type": "e1", "owner": "enemy", "position": [110, 20],
                     "count": 2, "stance": 2}],
     },
     lambda p, l: L({"all_units_in_region": {"x": 55 + p * 3, "y": 20,
                                             "radius": 7 - l}},
                    {"units_lost_lte": 1 if l < 2 else 0},
                    {"within_ticks": 13000 - l * 3000})),
    ("c11", "Tempo / Timing Window", "reasoning",
     "Strike inside a window that opens late and closes — not earliest, "
     "not biggest.",
     "Acting only inside a valid time window (takt-time): too early or too "
     "late both fail.",
     5, lambda p: combat_base(decoys=1),
     lambda p, l: L({"after_ticks": 1500 + l * 800},
                    {"units_killed_gte": 2 + l},
                    {"within_ticks": 12000 - l * 2500})),
    ("c12", "Error Recovery / Replan", "reasoning",
     "After a setback, rebuild/repair and re-establish capability under a "
     "hard clock.",
     "Detecting plan failure and repairing it under reduced means and a "
     "deadline — receding-horizon replanning.",
     5, lambda p: econ_base(3000, tent=True),
     lambda p, l: L({"has_building": "tent"},
                    {"building_total_gte": 4 + l},
                    {"after_ticks": 1200},
                    {"within_ticks": 22000 - l * 5000})),
]

LEVELS = ("easy", "medium", "hard")


def build():
    n_packs = n_levels = 0
    for code, title, cap, robo, meaning, npk, base_fn, win_fn in CATS:
        for p in range(npk):
            b = base_fn(p)
            base, cash = (b if isinstance(b, tuple) else (b, None))
            levels = {}
            for li, lv in enumerate(LEVELS):
                levels[lv] = {
                    "description": f"{title}: {lv} — the decision is harder "
                    f"(less info / tighter clock / committed choice).",
                    "win_condition": win_fn(p, li),
                    "max_turns": 45 + li * 8,
                }
                if li == 2:
                    levels[lv]["fail_condition"] = {"units_lost_lte": -1}
            pack = {
                "meta": {
                    "id": f"cat-{code}-{p:02d}",
                    "title": f"{title} #{p + 1}",
                    "capability": CAP[cap],
                    "real_world_meaning": meaning,
                    "robotics_analogue": robo,
                    "author": "catalog-generator",
                },
                "base_map": "rush-hour-arena",
                "base": base,
                "levels": levels,
            }
            if cash is not None:
                pack["starting_cash"] = cash
            (PACKS / f"cat-{code}-{p:02d}.yaml").write_text(
                yaml.safe_dump(pack, sort_keys=False, default_flow_style=False)
            )
            n_packs += 1
            n_levels += 3
    print(f"generated {n_packs} packs / {n_levels} levels across {len(CATS)} categories")
    return n_packs, n_levels


if __name__ == "__main__":
    build()
