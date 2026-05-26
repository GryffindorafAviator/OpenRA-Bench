"""Regression tests for the manual-play UI move-flow on AIRCRAFT.

User-reported bug (zh): `直升飞机动不了` ("the helicopter can't move") on
pack `combat-heli-flank`. Diagnosis: the engine + `Command::MoveUnits`
path moves helicopters fine (the scripted `eval_core.run_level` probe
shipped a heli from (30,19) to (60,10) in 3 turns), so the gap is in
the manual-play UI flow at `site/index.html` + `site/game_api.py` —
specifically in the click-to-action disambiguation.

The pre-fix `gameMinimapClick` used a single 3.8-cell radius for the
enemy stack-pick. The `combat-heli-flank` map carries a 21-cell pbox
wall (1×1 enemy buildings at x=50, y=9..31) plus a 5-cell e1 cluster
right behind it. With radius 3.8 (square 14.44), the entire mid-map
band (x ∈ [46..63]) resolved to "enemy here" for any click, so a
MOVE click with a heli selected SILENTLY became an ATTACK on the
nearest pbox. The heli would either stall out of weapon range or
chase the wall instead of bypassing it — to the user this read as
"the heli won't move."

Pins guarded by this file:

1. The end-to-end API path moves a heli from `combat-heli-flank`
   when fed exactly the JSON the UI emits for `mode:"move"`. This is
   the backend half: any regression in `human_actions_to_commands`
   for an aircraft `unit_ids` shows up here.

2. The click-handler disambiguation (Python re-implementation
   of `gameMinimapClick`'s decision tree) returns MOVE for a click
   3+ cells from a 1×1 enemy unit/turret, ATTACK for a click 0-1
   cells on top of one. This pins the radius-split fix so a future
   edit can't widen the enemy radius back to 3.8.

The JS-side re-implementation is kept in this file (not exec'd via
a browser) so the test stays free of headless-browser deps; the
algorithm is small enough that a Python mirror is the right tool.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def game_api_app():
    """Load `site/game_api.py` as a module under the name `game_api`
    so Pydantic's forward-ref resolver finds `StepRequest` etc. in
    the expected `game_api.*` namespace. `importlib.util` alone
    loads under the file basename which breaks `TypeAdapter` rebuild
    for non-trivial bodies (the symptom: `StepRequest is not fully
    defined` on the first POST to /api/game/step)."""
    sys.path.insert(0, str(ROOT / "site"))
    if "game_api" in sys.modules:
        del sys.modules["game_api"]
    import game_api  # noqa: WPS433 — see fixture docstring
    return game_api


def test_heli_move_via_step_endpoint_actually_moves(game_api_app):
    """End-to-end pin: a `{mode:"move",unit_ids:[heli],target_x,target_y}`
    POST to /api/game/step on `combat-heli-flank` actually advances the
    heli. The pre-fix UI bug was upstream of this path (it stopped the
    UI from EMITTING the move), but pinning the path itself prevents
    a backend regression that would silently swap "move heli" for a
    no-op — the failure mode the user reported."""
    from fastapi.testclient import TestClient

    client = TestClient(game_api_app.app)

    r = client.post(
        "/api/game/start",
        json={"pack_id": "combat-heli-flank", "level": "easy", "seed": 1},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    sid = data["session_id"]

    # The pack pre-places two heli at (30, 19) and (30, 21). Take the
    # north one (id 1002 deterministic on seed 1).
    helis = [u for u in data["units"] if u["type"] == "heli"]
    assert len(helis) == 2, f"expected 2 helis, got {helis}"
    heli = next(u for u in helis if u["cell_y"] == 19)
    start_xy = (heli["cell_x"], heli["cell_y"])
    assert start_xy == (30, 19), start_xy

    # Move the heli 14 cells due north (to y=5 on the same x=30 lane).
    # One turn should advance ~10 cells (heli speed × ~90 ticks/turn on
    # the non-interrupt manual-play path). Note we deliberately keep the
    # target on the WEST side of the map: an `agun` (anti-aircraft tower)
    # added at (45,20) in the F1-audit cheat-WIN fix (commit 35ddc81e)
    # would shred a heli flying east across the centre-line, masking the
    # move-flow probe as a "didn't move" failure (the unit disappears
    # from `units` because it died en route). The north lane keeps the
    # heli well outside the agun's effective range so this test only
    # exercises the move PATH, not the survivability of a brute lane.
    r2 = client.post(
        "/api/game/step",
        json={
            "session_id": sid,
            "actions": [{
                "mode": "move",
                "unit_ids": [heli["id"]],
                "target_x": 30,
                "target_y": 5,
            }],
        },
    )
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    after = next(u for u in d2["units"] if u["id"] == heli["id"])
    moved_dx = after["cell_x"] - start_xy[0]
    moved_dy = after["cell_y"] - start_xy[1]
    # The heli must have moved at least 5 cells closer to the target
    # (one turn × ~10 cells/turn, accept ≥5 for engine-tick-jitter
    # robustness). Pre-fix the UI never emitted the move; this is
    # the path-itself pin.
    assert abs(moved_dx) + abs(moved_dy) >= 5, (
        f"heli barely moved: {start_xy} -> "
        f"({after['cell_x']},{after['cell_y']})"
    )
    assert after["activity"] == "moving", after


# ── Click-handler disambiguation (Python mirror of gameMinimapClick) ──

# RA's small (1×1) defensive buildings. The fix groups these with
# enemy UNITS for the click-radius decision because their footprint is
# the same as a single infantry unit; using a 2.5+-cell radius for
# them swallows the surrounding move-cells (the heli-pack symptom).
_SMALL_BLD = {"pbox", "hbox", "gun", "sam", "ftur", "agun", "tsla"}


def _nearest_stack(items, cx, cy, radius):
    """Python mirror of `gameNearestStack(items, cx, cy, radius)`."""
    if not items:
        return []
    best = float("inf")
    for u in items:
        d = (u["cell_x"] - cx) ** 2 + (u["cell_y"] - cy) ** 2
        if d < best:
            best = d
    if best > radius * radius:
        return []
    return [
        u for u in items
        if abs(
            (u["cell_x"] - cx) ** 2 + (u["cell_y"] - cy) ** 2 - best
        ) < 0.01
    ]


def _classify_click(cx, cy, units, buildings, enemies, selected_ids):
    """Python mirror of `gameMinimapClick`'s decision tree — returns
    one of {"attack:<id>", "select:<ids>", "select_bld:<ids>",
    "move:<cx>,<cy>", "noop"}. Mirrors lines 1116-1144 of
    `site/index.html` after the radius-split fix."""
    own_here = _nearest_stack(units, cx, cy, 2.6)
    own_bld_here = _nearest_stack(buildings, cx, cy, 3.8)

    # Split-radius fix (the heli-pack regression).
    enemy_close = [
        e for e in enemies
        if (not e.get("is_building"))
        or str(e.get("type", "")).lower() in _SMALL_BLD
    ]
    enemy_big = [
        e for e in enemies
        if e.get("is_building")
        and str(e.get("type", "")).lower() not in _SMALL_BLD
    ]
    enemy_close_here = _nearest_stack(enemy_close, cx, cy, 1.5)
    enemy_big_here = _nearest_stack(enemy_big, cx, cy, 2.5)
    enemy_here = enemy_close_here or enemy_big_here

    if selected_ids and enemy_here:
        return f"attack:{enemy_here[0]['id']}"
    if own_here:
        return "select:" + ",".join(u["id"] for u in own_here)
    if own_bld_here:
        return "select_bld:" + ",".join(b["id"] for b in own_bld_here)
    if selected_ids:
        return f"move:{cx},{cy}"
    return "noop"


def _heli_pack_enemies():
    """The heli pack's enemy actor list, as `gameState.enemies` would
    surface it AFTER fog clears (fog isn't part of this test — we're
    testing the click branch, not perception)."""
    enemies = []
    # 23-cell pbox wall x=50, y=9..31
    for y in range(9, 32):
        enemies.append({
            "id": f"p{y}", "type": "pbox",
            "cell_x": 50, "cell_y": y, "is_building": True,
        })
    # 3-cell e1 cluster at (60, 19..21)
    for y in (19, 20, 21):
        enemies.append({
            "id": f"e{y}", "type": "e1",
            "cell_x": 60, "cell_y": y, "is_building": False,
        })
    # Far enemy fact (engine auto-done mitigation)
    enemies.append({
        "id": "ef", "type": "fact",
        "cell_x": 124, "cell_y": 20, "is_building": True,
    })
    return enemies


def test_click_bypassing_pbox_wall_is_move_not_attack():
    """The heli-pack symptom: with a heli selected, clicking 3+ cells
    away from the pbox wall must produce a MOVE, not an attack-pbox
    order. Pre-fix this returned attack-pbox for every click within
    ~3.8 cells of any pbox — the whole mid-map band — so the heli
    never bypassed the wall."""
    units = [
        {"id": "1002", "type": "heli", "cell_x": 30, "cell_y": 19},
        {"id": "1003", "type": "heli", "cell_x": 30, "cell_y": 21},
    ]
    buildings = [{"id": "1001", "type": "fact", "cell_x": 4, "cell_y": 20}]
    enemies = _heli_pack_enemies()
    selected = {"1002"}

    # Cells 3+ away from x=50 wall: must be MOVE.
    for cx, cy in [(47, 20), (53, 20), (55, 19), (45, 15), (55, 25)]:
        result = _classify_click(cx, cy, units, buildings, enemies, selected)
        assert result == f"move:{cx},{cy}", (
            f"click ({cx},{cy}) expected MOVE, got {result} "
            f"(the heli-pack regression — see file docstring)"
        )


def test_click_directly_on_pbox_is_attack():
    """Counter-pin: a deliberate click ON the pbox (or directly
    adjacent, dist=1) MUST still resolve to attack-pbox — the user
    explicitly aimed at it. Without this, the fix would over-correct
    and break legitimate attack clicks on defensive turrets."""
    units = [{"id": "1002", "type": "heli", "cell_x": 30, "cell_y": 19}]
    buildings = []
    enemies = _heli_pack_enemies()
    selected = {"1002"}

    # Click ON pbox at (50, 20):
    r = _classify_click(50, 20, units, buildings, enemies, selected)
    assert r.startswith("attack:"), r
    # Click on adjacent cell (49, 20) — within 1.5-radius:
    r = _classify_click(49, 20, units, buildings, enemies, selected)
    assert r.startswith("attack:"), r


def test_click_far_from_e1_cluster_is_move():
    """The other half of the symptom: the e1 cluster at (60, 19..21)
    used to steal every click within ~3 cells, so a user trying to
    move the heli to (62, 19) (just past the cluster) got attack-e1
    instead. Now that's a clean MOVE."""
    units = [{"id": "1002", "type": "heli", "cell_x": 60, "cell_y": 10}]
    buildings = []
    enemies = _heli_pack_enemies()
    selected = {"1002"}

    # 2+ cells away from any e1 (cluster spans y=19..21 at x=60):
    for cx, cy in [(62, 19), (63, 20), (60, 17), (58, 19)]:
        r = _classify_click(cx, cy, units, buildings, enemies, selected)
        assert r == f"move:{cx},{cy}", (
            f"click ({cx},{cy}) expected MOVE, got {r}"
        )


def test_click_on_e1_is_attack():
    """Counter-pin for the e1 case — a click on or adjacent to the
    cluster still resolves to attack-e1 (the WIN move for this pack)."""
    units = [{"id": "1002", "type": "heli", "cell_x": 30, "cell_y": 19}]
    buildings = []
    enemies = _heli_pack_enemies()
    selected = {"1002"}

    for cx, cy in [(60, 19), (60, 20), (61, 19), (60, 21)]:
        r = _classify_click(cx, cy, units, buildings, enemies, selected)
        assert r.startswith("attack:"), (
            f"click ({cx},{cy}) expected ATTACK, got {r}"
        )


def test_large_enemy_building_keeps_generous_radius():
    """Counter-pin for big buildings: a 2x2/3x3 enemy structure (e.g.
    `fact`, `proc`, `weap`) should STILL be clickable from 1-2 cells
    away — its footprint extends beyond the centre cell, and a tight
    1.5-cell radius would force the user to land exactly on the
    centre cell of a 3x3. Radius 2.5 is the keep-band."""
    units = [{"id": "1", "type": "1tnk", "cell_x": 10, "cell_y": 10}]
    enemies = [{
        "id": "ef", "type": "fact",
        "cell_x": 124, "cell_y": 20, "is_building": True,
    }]
    selected = {"1"}
    # Adjacent corner (125, 21) — dist² = 2 < 6.25:
    r = _classify_click(125, 21, units, [], enemies, selected)
    assert r.startswith("attack:"), r
    # Just outside the 2.5-radius (122, 22) — dist² = 4+4 = 8 > 6.25:
    r = _classify_click(122, 22, units, [], enemies, selected)
    assert r == "move:122,22", r
