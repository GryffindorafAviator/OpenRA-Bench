"""Smoke-test the spec-thief-steal-cash pack against three policies:

* stall          — observe-only → must LOSE (timeout)
* direct         — `infiltrate(thf, silo)` immediately → must LOSE
                   (the engine A*-routes through the pbox kill zone)
* flank          — `move_units` to a safe corner first, then
                   `infiltrate(thf, silo)` → must WIN

Runs all three policies × {easy, medium, hard} × seeds {1..4}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.eval_core import run_level


PACK = PACKS_DIR / "spec-thief-steal-cash.yaml"

# Flank waypoint (safe SE corner of the southern open zone, well
# clear of every pbox / gun turret) per level. Each is south of the
# wall and within Chebyshev > 6 from every defender.
FLANK_WAYPOINT = {
    "easy":   (3, 19),   # west flank, south of wall
    "medium": (3, 23),   # west flank, south of wall
    "hard":   (3, 27),   # west flank, deep south
}


def stall_policy(rs, Command):
    return [Command.observe()]


def direct_policy(rs, Command):
    """Try to infiltrate the silo immediately — engine A*-routes the
    thief through the short path (the defended direct gap)."""
    units = rs.get("units_summary") or []
    thieves = [u for u in units if str(u.get("type", "")).lower() == "thf"]
    if not thieves:
        return [Command.observe()]
    ebs = rs.get("enemy_buildings_summary") or []
    silo = next((b for b in ebs if str(b.get("type", "")).lower() == "silo"), None)
    if silo is None:
        return [Command.observe()]
    sid = str(silo["id"])
    # Pick the thief that doesn't yet have an Infiltrate activity
    return [
        Command.infiltrate([str(t["id"])], sid)
        for t in thieves
    ]


def make_flank_policy(level: str):
    """Two-phase: drive every thief through the west flank waypoint
    (south of the wall), then infiltrate."""
    wx, wy = FLANK_WAYPOINT[level]
    # Per-thief state across turns: 'phase' in {move, infiltrate}
    state: dict[str, str] = {}
    # Silos already targeted by another thief (so 2 thieves hit 2 silos)
    targeted: set[str] = set()

    def pol(rs, Command):
        units = rs.get("units_summary") or []
        thieves = [u for u in units if str(u.get("type", "")).lower() == "thf"]
        if not thieves:
            return [Command.observe()]
        ebs = rs.get("enemy_buildings_summary") or []
        silos = [b for b in ebs if str(b.get("type", "")).lower() == "silo"]
        if not silos:
            return [Command.observe()]
        cmds = []
        for t in thieves:
            tid = str(t["id"])
            tx, ty = int(t.get("cell_x", 0)), int(t.get("cell_y", 0))
            phase = state.get(tid, "move")
            if phase == "move":
                # Once thief is south of the wall on the west flank,
                # switch to infiltrate.
                if tx <= 5 and ty >= wy - 1:
                    phase = "infiltrate"
                    state[tid] = phase
            if phase == "move":
                cmds.append(Command.move_units([tid], wx, wy))
            else:
                # Pick a silo not already targeted by another thief
                free = [s for s in silos if str(s["id"]) not in targeted]
                target = (free or silos)[0]
                sid = str(target["id"])
                targeted.add(sid)
                cmds.append(Command.infiltrate([tid], sid))
        return cmds

    return pol


def run(level: str, policy, seed: int):
    c = compile_level(load_pack(PACK), level)
    return run_level(c, policy, seed=seed)


def main():
    levels = ["easy", "medium", "hard"]
    seeds = [1, 2, 3, 4]
    rows = []
    for lvl in levels:
        for name, pol in [
            ("stall", stall_policy),
            ("direct", direct_policy),
            ("flank", make_flank_policy(lvl)),
        ]:
            for sd in seeds:
                # Fresh closure for flank (state is per-episode)
                if name == "flank":
                    pol = make_flank_policy(lvl)
                res = run(lvl, pol, sd)
                rows.append({
                    "level": lvl,
                    "policy": name,
                    "seed": sd,
                    "outcome": res.outcome,
                    "turns": res.turns,
                    "cash": res.signals.cash,
                    "units_lost": res.signals.units_lost,
                    "tick": res.signals.game_tick,
                })
                print(
                    f"{lvl:7s} {name:7s} seed={sd}  "
                    f"outcome={res.outcome:5s}  turns={res.turns:3d}  "
                    f"cash={res.signals.cash:4d}  "
                    f"lost={res.signals.units_lost}  "
                    f"tick={res.signals.game_tick}"
                )
    # Summary: any mis-bar?
    print()
    print("=== bar check ===")
    fails = []
    for r in rows:
        if r["policy"] in ("stall", "direct") and r["outcome"] == "win":
            fails.append(("CHEAT WINS", r))
        if r["policy"] == "flank" and r["outcome"] != "win":
            fails.append(("INTENT LOSES", r))
    if fails:
        for tag, r in fails:
            print(f"  {tag}: {r}")
        return 1
    print("  ALL bar checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
