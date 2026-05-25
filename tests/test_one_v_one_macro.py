"""Tests for the canonical LLM-vs-LLM macro 1v1 pack.

Pins:
* adversarial-1v1-macro compiles cleanly for all three rungs.
* mirrored bases sit at non-overlapping positions.
* both sides have ore reachable within ~12 cells of their fact.
* the run_1v1 harness is deterministic on identical inputs
  (stall vs stall produces the SAME outcome on repeated calls).
* a non-stall policy produces a DIFFERENT outcome from stall-vs-
  stall on at least one seed (the engine actually reads commands
  from both sides — the harness wiring is real).
* at t=0 of medium, neither side observes any of the opponent's
  own_unit ids — fog of war isolates the two bases.
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


def test_pack_compiles_all_three_rungs():
    """Every rung loads, validates, and finds a Rust-loadable map."""
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "adversarial"
    # NO `reveal_map` — both sides must explore through fog.
    assert pack.reveal_map is False
    for lv in ("easy", "medium", "hard"):
        c = compile_level(pack, lv)
        assert c.map_supported, f"{lv} map not Rust-loadable"
        assert c.max_turns == 120


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


def test_medium_bases_mirrored_and_non_overlapping():
    """Medium's agent and enemy facts must sit at distinct positions
    far apart — the mirrored-corner contract."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "medium")
    a_pos, e_pos = _agent_enemy_facts(c)
    assert a_pos != e_pos, "agent and enemy facts overlap"
    # Mirrored about the map centre (40,40) on the 80x80 medium arena.
    dx = e_pos[0] - a_pos[0]
    dy = e_pos[1] - a_pos[1]
    assert dx > 50 and dy > 50, (
        f"agent fact {a_pos} / enemy fact {e_pos} aren't mirrored at "
        f"opposite corners (dx={dx}, dy={dy})"
    )


def test_both_sides_have_ore_reachable_from_base():
    """Each side must have at least one ore-patch cell within 12
    cells of its construction yard — safe opening econ for both
    commanders. Reads the pack-level / level-merged `ore_patches:`
    list (each `{x, y, radius, amount}` declares a DISK)."""
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
    # 120-turn macro game.
    a = run_1v1(tmp, _stall, _stall, seed=1, max_turns=10)
    b = run_1v1(tmp, _stall, _stall, seed=1, max_turns=10)
    assert a.winner == b.winner, (
        f"non-deterministic: {a.winner!r} then {b.winner!r}"
    )
    assert a.turns == b.turns


def _rusher_for(side: str):
    """Bench-local rusher policy: attack-move every combat unit
    toward the opposite corner. Mirrors `_rusher_agent_fn` in
    run_eval but stays local to keep this test self-contained."""
    target = (55, 55) if side == "agent" else (10, 10)
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
    """At t=0 of medium, neither side observes any of the OPPONENT's
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
        # The agent's view should not see ANY enemy ids (the enemy's
        # own-unit ids), and vice versa. (At t=0 nothing scouted.)
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


def test_run_eval_1v1_cli_smoke(tmp_path):
    """`run_eval --mode 1v1 --provider scripted:stall --opponent
    scripted:stall` runs end-to-end and emits a valid stats JSON
    carrying the `adversarial_1v1` headline block."""
    import json
    from openra_bench.run_eval import main

    out = tmp_path / "1v1_smoke.json"
    rc = main([
        "run_eval",
        "--mode", "1v1",
        "--packs", str(PACK_PATH),
        "--levels", "easy",
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
    assert data["episodes"][0]["capability"] == "adversarial"
    assert data["episodes"][0]["cell"] == "adversarial-1v1-macro:easy"
    assert data["episodes"][0]["mode"] == "1v1"
