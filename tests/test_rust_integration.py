"""Integration tests that boot the real Rust OpenRA engine.

These are *not* mocked: every test in `TestRustEngineTools` instantiates
`openra_train.OpenRAEnv` and drives it with rule-based bot players, then
asserts on observed engine behaviour (tool correctness + corner cases).
`TestStackIntegration` exercises the Bench stack (adapter, win
conditions, scenario packs, eval_core) on top of the live engine.

Behaviour pinned here was first probed against the engine, not assumed:
  * move_units drives a unit to (and reaching) the target cell
  * same (scenario, seed) is bit-for-bit deterministic
  * bad unit ids warn ("not owned"), don't raise
  * empty / invalid commands are safe no-ops
  * explored% and units_killed are monotonic non-decreasing

Run: pytest tests/test_rust_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

ot = pytest.importorskip("openra_train", reason="Rust env wheel not installed")

TRAIN = Path("/Users/berta/Projects/OpenRA-RL-Training")
RUSH_HOUR = str(TRAIN / "scenarios" / "discovery" / "rush-hour.yaml")

pytestmark = pytest.mark.skipif(
    not Path(RUSH_HOUR).exists(), reason="OpenRA-RL-Training scenarios not present"
)


# --------------------------------------------------------------------------- #
# Rule-based bot players                                                       #
# --------------------------------------------------------------------------- #
def idle_bot(_obs):
    """No-op: only `observe()`. Used to test determinism / no side effects."""
    return [ot.Command.observe()]


def charge_bot(target):
    """Move every owned unit toward a fixed cell. Tests move_units."""

    def _bot(obs):
        ids = list(obs.get("unit_positions", {}))
        if not ids:
            return [ot.Command.observe()]
        return [ot.Command.move_units(ids, target[0], target[1])]

    return _bot


def hunter_bot(obs):
    """Attack the first visible enemy; otherwise push east. Tests attack_unit
    and the discover→engage path."""
    ids = list(obs.get("unit_positions", {}))
    if not ids:
        return [ot.Command.observe()]
    enemies = obs.get("enemy_positions", []) or []
    if enemies:
        tid = str(enemies[0].get("id"))
        return [ot.Command.attack_unit(ids, tid)]
    return [ot.Command.move_units(ids, 120, 20)]


def _first_unit(obs):
    up = obs["unit_positions"]
    k = sorted(up)[0]
    return k, (up[k]["cell_x"], up[k]["cell_y"])


def _man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _run(env, bot, steps):
    """Drive `env` with `bot` for `steps`, yielding (obs, reward, done, info)."""
    obs = env.reset()
    out = []
    for _ in range(steps):
        o, r, d, i = env.step(bot(obs))
        out.append((o, r, d, i))
        obs = o
        if d:
            break
    return out


# --------------------------------------------------------------------------- #
# Engine + tool correctness (live Rust env)                                    #
# --------------------------------------------------------------------------- #
class TestRustEngineTools:
    def test_reset_schema_and_initial_state(self):
        obs = ot.OpenRAEnv(RUSH_HOUR, 7).reset()
        for key in (
            "unit_positions",
            "unit_hp",
            "enemy_positions",
            "explored_percent",
            "game_tick",
            "units_killed",
        ):
            assert key in obs, f"missing obs key {key!r}"
        assert obs["unit_positions"], "agent should own units at reset"
        # OpenRA reveals sight around starting units at game start, so
        # explored_percent must be > 0 at reset (engine fix) but well
        # below full-map coverage.
        assert 0.0 < obs["explored_percent"] < 100.0, obs["explored_percent"]
        assert obs["units_killed"] == 0
        assert obs["game_tick"] < 50

    def test_same_seed_is_deterministic(self):
        a, b = ot.OpenRAEnv(RUSH_HOUR, 11), ot.OpenRAEnv(RUSH_HOUR, 11)
        oa, ob = a.reset(), b.reset()
        assert oa["unit_positions"] == ob["unit_positions"]
        for _ in range(6):
            oa, *_ = a.step([ot.Command.observe()])
            ob, *_ = b.step([ot.Command.observe()])
        assert oa["unit_positions"] == ob["unit_positions"]
        assert oa["game_tick"] == ob["game_tick"]

    def test_move_units_drives_unit_to_target(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        obs = env.reset()
        uid, start = _first_unit(obs)
        # Target an in-bounds interior cell: rush-hour is ~128x40 and the
        # first unit may spawn at the east edge (x~120), so a blind +25
        # would be off-map (the engine correctly refuses to path off-map).
        tx = start[0] - 25 if start[0] > 64 else start[0] + 25
        target = (tx, min(max(start[1], 3), 36))
        last = start
        for _ in range(15):
            obs, *_ = env.step([ot.Command.move_units([uid], target[0], target[1])])
            last = (obs["unit_positions"][uid]["cell_x"], obs["unit_positions"][uid]["cell_y"])
        assert _man(last, target) < _man(start, target), "unit did not move toward target"
        assert _man(last, target) <= 1, f"unit {uid} did not reach {target}, at {last}"

    def test_idle_units_do_not_move(self):
        """A corner agent unit with no order must hold position (no
        spontaneous teleport / drift)."""
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        obs = env.reset()
        uid, start = _first_unit(obs)  # spawn-corner unit, far from enemies
        for _ in range(4):
            obs, *_ = env.step([ot.Command.observe()])
        pos = (obs["unit_positions"][uid]["cell_x"], obs["unit_positions"][uid]["cell_y"])
        assert pos == start, f"idle unit drifted {start} -> {pos}"

    def test_empty_command_list_is_safe(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        env.reset()
        obs, reward, done, info = env.step([])
        assert isinstance(done, bool) and done is False
        assert isinstance(reward, float)
        assert obs["game_tick"] > 0

    def test_invalid_unit_id_warns_not_raises(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        env.reset()
        _o, _r, _d, info = env.step([ot.Command.move_units(["999999"], 10, 10)])
        warns = info.get("warnings", [])
        assert any("999999" in w and "not owned" in w for w in warns), warns

    def test_invalid_attack_target_is_safe(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        obs = env.reset()
        uid, _ = _first_unit(obs)
        obs, _r, done, _i = env.step([ot.Command.attack_unit([uid], "888888")])
        assert done is False
        assert obs["unit_positions"], "engine state intact after bad attack target"

    def test_out_of_bounds_move_is_safe(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        obs = env.reset()
        uid, _ = _first_unit(obs)
        for _ in range(5):
            obs, _r, done, _i = env.step([ot.Command.move_units([uid], 99999, 99999)])
            assert done is False
        p = obs["unit_positions"][uid]
        assert 0 <= p["cell_x"] < 1000 and 0 <= p["cell_y"] < 1000

    def test_explored_percent_monotonic_and_grows(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        prev = -1.0
        seen = []
        for obs, *_ in _run(env, charge_bot((120, 20)), 20):
            assert obs["explored_percent"] >= prev - 1e-6, "explored% decreased"
            prev = obs["explored_percent"]
            seen.append(prev)
        assert seen[-1] > seen[0], "moving across the map revealed no new area"

    def test_units_killed_monotonic_nondecreasing(self):
        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        prev = 0
        for obs, *_ in _run(env, hunter_bot, 30):
            assert obs["units_killed"] >= prev, "units_killed went backwards"
            prev = obs["units_killed"]
        assert prev >= 0


# --------------------------------------------------------------------------- #
# Bench stack on the live engine                                               #
# --------------------------------------------------------------------------- #
class TestStackIntegration:
    def test_adapter_signals_track_engine(self):
        from openra_bench.rust_adapter import RustObsAdapter

        env = ot.OpenRAEnv(RUSH_HOUR, 7)
        ad = RustObsAdapter()
        ad.observe(env.reset())
        bot = charge_bot((120, 20))
        prev_seen = 0
        for _ in range(20):
            o, _r, d, _i = env.step(bot({"unit_positions": ad._raw.get("unit_positions", {})}))
            ad.observe(o, done=d)
            # discovery set is cumulative — never shrinks
            assert len(ad.signals.enemies_seen_ids) >= prev_seen
            prev_seen = len(ad.signals.enemies_seen_ids)
        kw = ad.signals.as_reward_kwargs()
        assert set(kw) >= {
            "units_killed",
            "units_lost",
            "explored_percent",
            "enemies_discovered",
            "outcome",
            "game_tick",
            "done",
        }
        assert kw["explored_percent"] > 0.0
        assert kw["units_lost"] >= 0

    def test_win_condition_predicates_pure(self):
        from openra_bench.rust_adapter import EpisodeSignals
        from openra_bench.scenarios.win_conditions import WinContext, evaluate

        sig = EpisodeSignals(explored_percent=60.0, units_killed=3, game_tick=4000)
        sig.enemies_seen_ids = {"a", "b"}
        rs = {"units_summary": [{"id": "1", "cell_x": 10, "cell_y": 10, "type": None}]}
        ctx = WinContext(signals=sig, render_state=rs)

        assert evaluate({"explored_pct_gte": 50}, ctx) is True
        assert evaluate({"explored_pct_gte": 75}, ctx) is False
        assert evaluate({"enemies_discovered_gte": 2}, ctx) is True
        assert evaluate({"within_ticks": 5000}, ctx) is True
        assert evaluate({"within_ticks": 3000}, ctx) is False
        assert evaluate({"reach_region": {"x": 11, "y": 11, "radius": 3}}, ctx) is True
        assert evaluate({"reach_region": {"x": 99, "y": 99, "radius": 2}}, ctx) is False
        # composites
        assert evaluate({"all_of": [{"explored_pct_gte": 50}, {"within_ticks": 5000}]}, ctx)
        assert evaluate({"any_of": [{"explored_pct_gte": 99}, {"units_killed_gte": 3}]}, ctx)
        assert evaluate({"not": {"explored_pct_gte": 99}}, ctx) is True

    def test_win_condition_rejects_unknown_keys(self):
        from openra_bench.scenarios.win_conditions import WinCondition

        with pytest.raises(ValueError):
            WinCondition(definitely_not_a_predicate=1)
        with pytest.raises(ValueError):
            WinCondition(all_of=[{"explored_pct_gte": 1}], explored_pct_gte=2)

    def test_eval_core_win_and_loss_wiring_deterministic(self):
        """Pin the win/fail plumbing without depending on bot skill:
        a trivially-true win condition => 'win' on turn 1; a trivially-
        true fail condition => 'loss'."""
        from openra_bench.eval_core import run_level
        from openra_bench.scenarios.schema import ScenarioPack

        base = {
            "agent": {"faction": "allies"},
            "enemy": {"faction": "soviet"},
            "tools": ["move_units", "attack_unit"],
            "actors": [
                {"type": "jeep", "owner": "agent", "position": [5, 5], "count": 2},
                {"type": "e1", "owner": "enemy", "position": [60, 20], "stance": 2},
            ],
            "termination": {"max_ticks": 8000},
        }
        meta = {
            "id": "selftest-wiring",
            "title": "Self Test",
            "capability": "action",
            "real_world_meaning": "deterministic plumbing check for the runner",
            "robotics_analogue": "unit-test harness",
            "author": "ci",
        }

        def pack(win, fail=None):
            lvl = {
                "description": "wiring check level",
                "overrides": {},
                "win_condition": win,
                "max_turns": 3,
            }
            if fail:
                lvl["fail_condition"] = fail
            return ScenarioPack(
                meta=meta, base_map="rush-hour-arena", base=base,
                levels={"easy": lvl, "medium": lvl, "hard": lvl},
            )

        win_pack = pack({"after_ticks": 0})  # always true once ticks >= 0
        res = run_level(win_pack.compile("easy"), bot_to_agent(idle_bot), seed=3)
        assert res.outcome == "win"
        assert res.signals.outcome == 1.0
        assert res.turns == 1
        assert len(res.trace) == res.turns

        loss_pack = pack({"explored_pct_gte": 999}, fail={"after_ticks": 0})
        res2 = run_level(loss_pack.compile("easy"), bot_to_agent(idle_bot), seed=3)
        assert res2.outcome == "loss"
        assert res2.signals.outcome == 0.0

    @pytest.mark.parametrize(
        "pack_file,level",
        [
            ("perception-frontier-reading.yaml", "easy"),
            ("reasoning-frontier-commit.yaml", "easy"),
            ("action-multiunit-coordination.yaml", "easy"),
        ],
    )
    def test_authored_packs_run_end_to_end(self, pack_file, level):
        from openra_bench.eval_core import run_level, scripted_explore_agent
        from openra_bench.scenarios import load_pack
        from openra_bench.scenarios.loader import PACKS_DIR, compile_level

        pack = load_pack(PACKS_DIR / pack_file)
        compiled = compile_level(pack, level)
        assert compiled.map_supported, "authored packs must target a Rust-loadable map"
        res = run_level(compiled, scripted_explore_agent, seed=1)
        assert res.outcome in {"win", "draw", "loss"}
        assert 1 <= res.turns <= compiled.max_turns
        assert len(res.trace) == res.turns
        assert res.signals.game_tick > 0
        assert res.signals.as_reward_kwargs()["outcome"] in (0.0, 0.5, 1.0)


def bot_to_agent(bot):
    """Adapt an obs-only rule bot to the eval_core agent_fn signature
    `(render_state, Command) -> [Command]`. The render_state carries
    unit ids the bot needs under units_summary."""

    def _agent(render_state, Command):
        up = {
            str(u["id"]): {"cell_x": u["cell_x"], "cell_y": u["cell_y"]}
            for u in render_state.get("units_summary", [])
        }
        return bot({"unit_positions": up, "enemy_positions": []})

    return _agent
