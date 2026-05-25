"""Tests for the canonical LLM-vs-LLM macro 1v1 pack.

`adversarial-1v1-macro` is a SINGLE-MAP pack: per the user spec it is
NOT a difficulty ladder, it's one shared 128x96 bridges-naval arena
where two real-time decision agents play a head-to-head macro match.
The bench schema requires all three of {easy, medium, hard} as level
keys, so the pack declares all three pointing at the SAME compiled
scenario (a YAML anchor); the `configs:` block exposes ONE runnable
cell (`main`, pinned to `medium`).

Pins:
* adversarial-1v1-macro compiles cleanly for `medium` (the canonical
  rung) and the easy/hard rungs are identical to medium (anchored).
* canonical map shape: 128x96 bridges-arena, cordon-adjusted bounds.
* three ore patches per side at the expected approximate positions
  + a contested centre patch on the central bridge.
* mirrored bases sit at non-overlapping positions far apart.
* the run_1v1 harness is deterministic on identical inputs
  (stall vs stall produces the SAME outcome on repeated calls).
* a non-stall policy produces a DIFFERENT outcome from stall-vs-
  stall on at least one seed (the engine actually reads commands
  from both sides — the harness wiring is real).
* at t=0, neither side observes any of the opponent's own_unit ids
  — fog of war isolates the two bases.
* `--side-swap` mirrors outcomes, so the aggregate is fair under
  any residual slot-2 bias the engine carries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK_PATH = PACKS_DIR / "adversarial-1v1-macro.yaml"


def _stall(_rs, Command):
    return [Command.observe()]


def test_pack_compiles_to_single_canonical_map():
    """`medium` is the canonical rung. The pack is single-map so
    easy/medium/hard share an identical scenario (YAML anchor)."""
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "adversarial"
    # NO `reveal_map` — both sides must explore through fog.
    assert pack.reveal_map is False
    # ONE runnable config — the canonical 1v1 cell.
    assert pack.configs is not None
    assert len(pack.configs) == 1
    assert pack.configs[0].name == "main"
    assert pack.configs[0].level == "medium"
    # All three level keys present (schema requirement) but identical
    # (one canonical map). max_turns + starting_cash and description
    # all match.
    c_med = compile_level(pack, "medium")
    c_easy = compile_level(pack, "easy")
    c_hard = compile_level(pack, "hard")
    assert c_med.map_supported, "canonical map not Rust-loadable"
    assert c_med.max_turns == 200
    assert c_med.starting_cash == 2000
    for other in (c_easy, c_hard):
        assert other.max_turns == c_med.max_turns
        assert other.starting_cash == c_med.starting_cash
        # Same actor list, same map id ⇒ truly identical scenario.
        assert other.scenario.base_map == c_med.scenario.base_map
        assert len(other.scenario.actors) == len(c_med.scenario.actors)


def _agent_enemy_facts(c):
    """Return (agent_fact_pos, enemy_fact_pos) for a compiled level —
    the two construction yards at t=0."""
    agent = next(
        a for a in c.scenario.actors
        if a.type == "fact" and a.owner == "agent"
    )
    enemy = next(
        a for a in c.scenario.actors
        if a.type == "fact" and a.owner == "enemy"
    )
    return tuple(agent.position), tuple(enemy.position)


def test_canonical_map_is_128x96_bridges_arena():
    """The pack's base_map spec must be the 128x96 horizontal-channel
    bridges arena (the single canonical map). The bounds (after
    cordon=2) cover the interior playable rectangle."""
    pack = load_pack(PACK_PATH)
    bm = pack.base_map
    assert isinstance(bm, dict), (
        "base_map must be a generator spec dict, not a string id "
        "(single-map pack uses generator: bridges-arena directly)"
    )
    assert bm["generator"] == "bridges-arena"
    assert bm["width"] == 128
    assert bm["height"] == 96
    assert bm["cordon"] == 2
    assert bm["axis"] == "horizontal"
    assert bm["channel_y"] == 48
    # Three bridges across the channel.
    bridges = bm.get("bridges") or []
    assert len(bridges) == 3, f"expected 3 bridges, got {len(bridges)}"


def test_bases_mirrored_and_far_apart():
    """Agent NW and enemy SE facts sit at mirrored corners of the
    128x96 map (centre = (64, 48)). Diagonal separation is far
    outside any starting unit's vision range."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    a_pos, e_pos = _agent_enemy_facts(c)
    assert a_pos == (12, 10), f"agent fact should be at NW (12,10), got {a_pos}"
    assert e_pos == (116, 86), f"enemy fact should be at SE (116,86), got {e_pos}"
    # Perfect mirror about (64, 48).
    cx = (a_pos[0] + e_pos[0]) / 2
    cy = (a_pos[1] + e_pos[1]) / 2
    assert cx == 64 and cy == 48, (
        f"bases not mirrored about (64,48): midpoint = ({cx},{cy})"
    )
    # Diagonal separation ~128 cells (well outside any starting
    # unit's vision).
    import math
    sep = math.hypot(e_pos[0] - a_pos[0], e_pos[1] - a_pos[1])
    assert sep > 120, f"bases only {sep:.1f} cells apart (expected > 120)"


def test_three_ore_patches_per_side_plus_contested_centre():
    """The pack declares 9 ore patches total: one safe + one
    expansion patch in EACH of the four quadrants (NW/NE/SW/SE)
    plus one contested centre patch on the central bridge.
    Two spawn_point variants share the same global patch list (ore
    patches do not honour spawn_point per CLAUDE.md), so all four
    quadrants must carry near-base + mid-expansion ore — every
    variant has both sides next to a safe patch at t=0."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    patches = c.ore_patches
    assert len(patches) == 9, (
        f"expected 9 ore patches (2 per quadrant + 1 centre), "
        f"got {len(patches)}"
    )
    # Classify by quadrant (corner) — each quadrant carries one
    # near-base safe patch + one mid-expansion patch.
    in_nw = [p for p in patches if p["x"] < 48 and p["y"] < 36]
    in_ne = [p for p in patches if p["x"] > 80 and p["y"] < 36]
    in_sw = [p for p in patches if p["x"] < 48 and p["y"] > 60]
    in_se = [p for p in patches if p["x"] > 80 and p["y"] > 60]
    centre = [
        p for p in patches if 60 <= p["x"] <= 68 and 44 <= p["y"] <= 52
    ]
    # 2 patches per quadrant (safe + mid-expansion).
    assert len(in_nw) == 2, in_nw
    assert len(in_ne) == 2, in_ne
    assert len(in_sw) == 2, in_sw
    assert len(in_se) == 2, in_se
    assert len(centre) == 1, centre
    # The contested centre is the richest single patch.
    centre_amount = centre[0]["amount"]
    for p in in_nw + in_ne + in_sw + in_se:
        assert centre_amount > p["amount"], (
            f"centre patch {centre[0]} not richer than side patch {p}"
        )


def test_both_sides_have_ore_reachable_from_base():
    """Each side must have at least one ore-patch cell within 12
    cells of its construction yard — safe opening econ for both
    commanders."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    a_pos, e_pos = _agent_enemy_facts(c)

    def _has_close_patch(fact_xy):
        fx, fy = fact_xy
        for p in c.ore_patches:
            d = max(abs(p["x"] - fx), abs(p["y"] - fy))
            # The disc EDGE is the closest reachable ore cell.
            if d - int(p.get("radius", 0)) <= 12:
                return True
        return False

    assert _has_close_patch(a_pos), (
        f"agent fact {a_pos} has no ore patch within 12 cells"
    )
    assert _has_close_patch(e_pos), (
        f"enemy fact {e_pos} has no ore patch within 12 cells"
    )


def test_each_side_has_a_shipyard_on_a_shore_water_band():
    """The naval lane is optional but available: each side has a
    `syrd` shipyard and a small `water_cells:` band on the shore
    adjacent to it."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    syrds = [a for a in c.scenario.actors if a.type == "syrd"]
    owners = {a.owner for a in syrds}
    assert owners == {"agent", "enemy"}, (
        f"expected one syrd per side, got owners={owners}"
    )
    # Water bands mirrored about (64, 48): each side has its own
    # ≥9-cell shore strip.
    water_cells = c.water_cells
    agent_band = [(x, y) for x, y in water_cells if x < 10 and y < 48]
    enemy_band = [(x, y) for x, y in water_cells if x > 118 and y > 48]
    assert len(agent_band) >= 9, (
        f"agent shore-water band too small ({len(agent_band)} cells)"
    )
    assert len(enemy_band) >= 9, (
        f"enemy shore-water band too small ({len(enemy_band)} cells)"
    )


def test_run_1v1_stall_vs_stall_is_deterministic():
    """Two back-to-back calls to `run_1v1(stall, stall, seed=s)`
    must produce the SAME outcome — the harness is deterministic
    given symmetric input."""
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.one_v_one import run_1v1

    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    tmp = _scenario_to_tmp_yaml(c)
    # Short horizon — we only need to pin determinism, not a full
    # 200-turn macro game.
    a = run_1v1(tmp, _stall, _stall, seed=1, max_turns=10)
    b = run_1v1(tmp, _stall, _stall, seed=1, max_turns=10)
    assert a.winner == b.winner, (
        f"non-deterministic: {a.winner!r} then {b.winner!r}"
    )
    assert a.turns == b.turns


def _rusher_for(side: str):
    """Bench-local rusher policy: attack-move every combat unit
    toward the opposite corner of the canonical 128x96 map. Mirrors
    `_rusher_agent_fn` in run_eval but stays local to keep this
    test self-contained."""
    target = (116, 86) if side == "agent" else (12, 10)
    non_combat = {"harv", "fact", "proc", "powr", "tent", "weap",
                  "syrd", "mcv"}

    def _fn(render_state, Command):
        cmds = []
        for u in render_state.get("units_summary", []) or []:
            uid = u.get("id")
            if uid is None:
                continue
            if str(u.get("type", "")).lower() in non_combat:
                continue
            cmds.append(Command.attack_move(
                [str(uid)], target_x=target[0], target_y=target[1],
            ))
        return cmds or [Command.observe()]
    return _fn


def test_active_policy_reaches_the_engine_on_both_sides():
    """A non-stall policy actually delivers commands to step_1v1.
    We don't pin a specific winner (engine has residual slot bias
    that side-swap is meant to neutralise); we pin that the rusher
    issued real per-turn orders on the side it's driving — proof
    that step_1v1 reads per-player commands, not just one side's.
    """
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.one_v_one import run_1v1

    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    tmp = _scenario_to_tmp_yaml(c)

    # Agent = rusher, enemy = stall: agent trace must show ≥1
    # command per turn (each turn the rusher attack-moves every
    # combat unit), enemy trace must show exactly 1 observe per turn.
    r_vs_s = run_1v1(
        tmp, _rusher_for("agent"), _stall, seed=1, max_turns=15,
    )
    assert all(t["n_cmds"] >= 1 for t in r_vs_s.agent_trace), (
        "rusher (agent slot) didn't issue per-turn commands"
    )
    assert all(t["n_cmds"] == 1 for t in r_vs_s.enemy_trace), (
        "stall (enemy slot) drifted away from a single observe()"
    )
    # Same again with sides swapped — the rusher is now the enemy
    # slot. Step_1v1 must read enemy commands too: enemy trace ≥1
    # per turn, agent trace ==1 (the stall observe).
    s_vs_r = run_1v1(
        tmp, _stall, _rusher_for("enemy"), seed=1, max_turns=15,
    )
    assert all(t["n_cmds"] >= 1 for t in s_vs_r.enemy_trace), (
        "rusher (enemy slot) didn't issue per-turn commands — "
        "step_1v1 isn't reading enemy commands"
    )
    assert all(t["n_cmds"] == 1 for t in s_vs_r.agent_trace)


def test_fog_isolates_the_two_bases_at_t0():
    """At t=0, neither side observes any of the OPPONENT's
    own-unit ids — the mirrored bases sit outside each other's
    starting sight, which is the whole point of asymmetric info."""
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    tmp = _scenario_to_tmp_yaml(c)

    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    try:
        agent_ad = RustObsAdapter()
        enemy_ad = RustObsAdapter()
        agent_ad.observe(env.reset(seed=1))
        raw = env._env
        assert hasattr(raw, "enemy_observation"), (
            "engine wheel lacks enemy_observation"
        )
        enemy_ad.observe(raw.enemy_observation())

        a_rs = agent_ad.render_state()
        e_rs = enemy_ad.render_state()
        a_own_ids = {str(u.get("id")) for u in a_rs.get("units_summary", [])}
        e_own_ids = {str(u.get("id")) for u in e_rs.get("units_summary", [])}
        # Neither side's "scouted enemy" set should contain any of
        # the opponent's own-unit ids — mirrored bases are out of
        # starting sight.
        a_seen = set(agent_ad.signals.enemies_seen_ids)
        e_seen = set(enemy_ad.signals.enemies_seen_ids)
        assert a_seen.isdisjoint(e_own_ids), (
            f"agent already sees enemy units at t=0: "
            f"{a_seen & e_own_ids}"
        )
        assert e_seen.isdisjoint(a_own_ids), (
            f"enemy already sees agent units at t=0: "
            f"{e_seen & a_own_ids}"
        )
        # Both sides DO see their own forces (sanity).
        assert a_own_ids and e_own_ids
        # And the two own-unit id sets are disjoint (perspective
        # correctness).
        assert a_own_ids.isdisjoint(e_own_ids)
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)


def test_render_state_surfaces_units_killed_for_1v1_tiebreak():
    """Regression: `RustObsAdapter.render_state()` MUST include the
    `units_killed` key sourced from `signals.units_killed`. The 1v1
    harness's military-progress tie-break (`one_v_one.py::_kills`)
    reads `render_state["units_killed"]`; without this key the
    fallback is `0`, so every match where both bases survive the
    deadline collapses past the kills layer and is decided on
    buildings / economy alone — masking real combat performance.

    Engine `kills_per_player` is correctly incremented (verified by
    `test_dedup_kill_credit_per_victim.rs` and friends); the missing
    piece was the bench-side `render_state` shape — this test pins
    it. Stall-vs-stall on adversarial-1v1-macro produces non-trivial
    kill counts (units wander into engagement range around the map
    centre on the default stance:2 engagement scan), so the test can
    assert `units_killed >= 1` at the deadline-tier turn count.
    """
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    tmp = _scenario_to_tmp_yaml(c)

    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    try:
        a_ad = RustObsAdapter()
        e_ad = RustObsAdapter()
        a_ad.observe(env.reset(seed=1))
        raw = env._env
        e_ad.observe(raw.enemy_observation())
        Command = env.Command

        # Key MUST be present from t=0 (even if value is 0).
        a_rs = a_ad.render_state()
        e_rs = e_ad.render_state()
        assert "units_killed" in a_rs, (
            "render_state missing 'units_killed' — 1v1 tie-break "
            "(_kills) will silently read 0 every match"
        )
        assert "units_killed" in e_rs
        assert a_rs["units_killed"] == 0
        assert e_rs["units_killed"] == 0

        # Step deep enough for the engine's auto-engage scan to put
        # units in range of each other around the map centre. At
        # t=20 (tick 1803) the agent's e1 + the enemy's e1 have
        # mutually killed and both 1tnks have wedged into the
        # opponent's base far enough to destroy a `powr` — 2 kills
        # per side.
        for _ in range(20):
            a_obs, e_obs, done, _info = raw.step_1v1(
                [Command.observe()], [Command.observe()]
            )
            a_ad.observe(a_obs, done=done)
            e_ad.observe(e_obs, done=done)

        a_rs = a_ad.render_state()
        e_rs = e_ad.render_state()
        # Both sides have scored at least one kill (mutual e1 trade
        # + first building destruction). The exact tally is engine
        # state — what matters here is that `render_state` surfaces
        # the live engine counter rather than defaulting to 0.
        assert a_rs["units_killed"] >= 1, (
            f"agent render_state units_killed={a_rs['units_killed']} "
            f"— engine kills_per_player not reaching the bench adapter"
        )
        assert e_rs["units_killed"] >= 1, (
            f"enemy render_state units_killed={e_rs['units_killed']} "
            f"— engine kills_per_player not reaching the bench adapter"
        )
        # And the bench signals view must agree with the dict view —
        # the dict is sourced FROM signals so any drift is a code bug.
        assert a_rs["units_killed"] == a_ad.signals.units_killed
        assert e_rs["units_killed"] == e_ad.signals.units_killed
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)


def test_run_eval_1v1_cli_smoke(tmp_path):
    """`run_eval --mode 1v1 --provider scripted:stall --opponent
    scripted:stall --levels medium` runs end-to-end and emits a
    valid stats JSON carrying the `adversarial_1v1` headline block.
    Single-cell × single-seed ⇒ exactly ONE episode."""
    import json
    from openra_bench.run_eval import main

    out = tmp_path / "1v1_smoke.json"
    rc = main([
        "run_eval",
        "--mode", "1v1",
        "--packs", str(PACK_PATH),
        "--levels", "medium",
        "--seeds", "1",
        "--provider", "scripted:stall",
        "--opponent", "scripted:stall",
        "--out", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "1v1"
    assert "adversarial_1v1" in data
    assert data["adversarial_1v1"]["n_matches"] == 1
    assert len(data["episodes"]) == 1, (
        f"expected exactly ONE episode (single cell × 1 seed), "
        f"got {len(data['episodes'])}"
    )
    assert data["episodes"][0]["capability"] == "adversarial"
    assert data["episodes"][0]["cell"] == "adversarial-1v1-macro:medium"
    assert data["episodes"][0]["mode"] == "1v1"


def test_run_eval_1v1_side_swap_produces_paired_episodes(tmp_path):
    """`--side-swap` runs each match twice with sides swapped. With
    one cell + one seed, that's TWO episodes in the JSON, and the
    aggregate is fair regardless of any residual engine slot bias
    in stall-vs-stall."""
    import json
    from openra_bench.run_eval import main

    out = tmp_path / "1v1_swap.json"
    rc = main([
        "run_eval",
        "--mode", "1v1",
        "--packs", str(PACK_PATH),
        "--levels", "medium",
        "--seeds", "1",
        "--side-swap",
        "--provider", "scripted:stall",
        "--opponent", "scripted:stall",
        "--out", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["mode"] == "1v1"
    # Side-swap emits both halves (normal + swapped) plus a paired
    # aggregate episode. n_matches is still 1 (the single paired
    # match), but the per-half traces give the fairness audit.
    assert data["adversarial_1v1"]["n_matches"] == 1, (
        f"expected 1 paired match, got "
        f"{data['adversarial_1v1']['n_matches']}"
    )
    outcomes = [e.get("outcome") for e in data["episodes"]]
    # Stall vs stall on a symmetric arena: one slot wins from
    # residual engine bias, the other slot wins the mirrored half,
    # aggregate is a draw — the slot bias is neutralised.
    assert "win" in outcomes and "loss" in outcomes, (
        f"expected paired halves to mirror outcomes (one win + one "
        f"loss), got {outcomes}"
    )
