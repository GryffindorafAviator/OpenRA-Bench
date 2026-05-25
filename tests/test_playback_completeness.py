"""Playback must capture everything needed to *replay reasoning*.

Three gaps closed here, each asserted end to end:

1. Model reasoning/thinking is extracted from the provider raw reply
   and persisted on the assistant turn (so messages.json contains the
   chain-of-thought, like the training traces) — while the *wire*
   messages stay OpenAI-clean (no non-standard keys leak back).
2. Every turn records a goal tracker: per-win-condition-leaf progress
   AND a normalized cumulative reward vector, side by side.
3. The saved episode is loadable by the viewer's data layer.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.goal_tracker import leaf_progress, reward_vector, turn_goal
from openra_bench.providers import ChatReply, OpenAICompatibleProvider, ProviderConfig
from openra_bench.scenarios.win_conditions import WinContext


# ---- 1. reasoning capture --------------------------------------------------


class _Sig:
    explored_percent = 40.0
    enemies_seen_ids = ["e1", "e2"]
    enemy_buildings_seen_ids: list[str] = []
    units_killed = 3
    units_lost = 1
    game_tick = 1200
    cash = 1400
    resources = 600
    power_provided = 100
    power_drained = 60
    own_building_types = {"powr"}
    own_buildings = [("powr", 5, 5)]


def test_provider_extracts_reasoning_content():
    cfg = ProviderConfig(provider="vllm")
    prov = OpenAICompatibleProvider(cfg)
    data = {
        "choices": [
            {
                "message": {
                    "content": "moving out",
                    "reasoning_content": "enemy is east, scout first",
                    "tool_calls": [],
                }
            }
        ]
    }
    reply = prov._reply_from_data(data)  # pure parse, no network
    assert isinstance(reply, ChatReply)
    assert reply.reasoning == "enemy is east, scout first"
    # also accepts the OpenRouter-style flat "reasoning" key
    data["choices"][0]["message"] = {"content": "x", "reasoning": "because"}
    assert prov._reply_from_data(data).reasoning == "because"


def test_wire_messages_strip_nonstandard_keys():
    # history may carry reasoning for playback; it must NOT be posted.
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a", "reasoning": "secret",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "observe", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
    ]
    wire = OpenAICompatibleProvider._wire_messages(msgs)
    assert all("reasoning" not in m for m in wire)
    # structural keys survive
    assert wire[1]["tool_calls"][0]["function"]["name"] == "observe"
    assert wire[2]["tool_call_id"] == "c0"
    # original untouched (pure)
    assert msgs[1]["reasoning"] == "secret"


def test_agent_records_reasoning_on_assistant_turn():
    from openra_bench.agent import ModelAgent

    class P:
        def complete(self, messages, tools):
            return ChatReply(text="go", tool_calls=[], reasoning="I think east")

    a = ModelAgent(ProviderConfig(vision=False), allowed_tools=["observe"],
                   provider=P())
    a.agent_fn({"minimap": "", "units_summary": [], "enemy_summary": []},
               type("C", (), {"observe": staticmethod(lambda: "obs")}))
    asst = [m for m in a.history if m["role"] == "assistant"][-1]
    assert asst["reasoning"] == "I think east"


# ---- 2. per-turn goal tracker ---------------------------------------------


def _ctx():
    return WinContext(signals=_Sig(), render_state={"units_summary": []})


def test_leaf_progress_reports_ratio_per_predicate():
    wc = {"all_of": [{"units_killed_gte": 5}, {"explored_pct_gte": 80},
                     {"cash_gte": 1000}]}
    prog = leaf_progress(wc, _ctx())
    by = {p["name"]: p for p in prog}
    assert by["units_killed_gte"]["current"] == 3
    assert by["units_killed_gte"]["target"] == 5
    assert by["units_killed_gte"]["ratio"] == pytest.approx(0.6)
    assert by["cash_gte"]["satisfied"] is True  # 1400 >= 1000
    assert by["explored_pct_gte"]["satisfied"] is False


def test_reward_vector_is_normalized_and_cumulative():
    rv = reward_vector(_Sig())
    for k in ("economy", "military", "territory", "scouting", "objective"):
        assert k in rv and 0.0 <= rv[k] <= 1.0
    assert rv["territory"] == pytest.approx(0.40)  # 40% explored


def test_turn_goal_bundles_predicates_and_vector_side_by_side():
    g = turn_goal({"units_killed_gte": 5}, _ctx())
    assert "leaves" in g and "reward_vector" in g
    # `leaves_final` shape (per-leaf source of truth) is non-empty and
    # carries the current/target/satisfied fields the renderer relies on.
    assert g["leaves"]
    leaf = g["leaves"][0]
    assert leaf["name"] == "units_killed_gte"
    assert leaf["target"] == 5
    assert leaf["current"] == 3
    assert leaf["satisfied"] is False
    # The blocking-ratio scalar replaces the old mean-of-leaves
    # `objective_progress`; both keys carry the same min-of-leaves
    # value during the one-release deprecation window.
    assert "objective_blocking_ratio" in g
    assert 0.0 <= g["objective_blocking_ratio"] <= 1.0
    assert g["objective_blocking_ratio"] == g["objective_progress"]
    assert g["won"] is False  # 3 < 5


def test_turn_goal_blocking_ratio_is_min_not_mean():
    """Anti-regression for the misleading-average defect.

    `units_killed_gte:7` with 4 kills (ratio 0.57) + `within_ticks:4000`
    at tick 4203 (a *violated* deadline — `satisfied=False`, ratio
    softly clamped to ~0.95) used to produce `mean(0.57, 0.95) ≈ 0.76`
    — reading "near win" when both clauses had FAILED. The fix:
    `min(0.57, 0.95) = 0.57` — the worst leaf is the bottleneck
    constraint of an `all_of`, and the scalar refuses to inflate
    past it. Both leaves are also reported `satisfied=False`, which
    is the load-bearing flag downstream consumers (`won` derivation,
    leaf table renderer) check directly.
    """

    class _S:
        explored_percent = 0.0
        enemies_seen_ids: list = []
        enemy_buildings_seen_ids: list = []
        units_killed = 4
        units_lost = 0
        game_tick = 4203
        cash = 0
        resources = 0
        power_provided = 0
        power_drained = 0
        own_building_types: set = set()
        own_buildings: list = []

    ctx = WinContext(signals=_S(), render_state={"units_summary": []})
    wc = {"all_of": [{"units_killed_gte": 7}, {"within_ticks": 4000}]}
    g = turn_goal(wc, ctx)
    by = {leaf["name"]: leaf for leaf in g["leaves"]}
    # The kills leaf is the bottleneck: ratio 4/7 ≈ 0.571.
    assert by["units_killed_gte"]["ratio"] == pytest.approx(4 / 7, abs=1e-4)
    assert by["units_killed_gte"]["satisfied"] is False
    # within_ticks: deadline missed — satisfied=False — but the soft
    # ratio is still > 0 (≈ 4000/4203). The point of the new scalar
    # is that the MEAN of these (~0.76) would mislead; the MIN
    # honestly reports the worst leaf.
    assert by["within_ticks"]["satisfied"] is False
    assert by["within_ticks"]["ratio"] > 0.9
    # New scalar: the WORST leaf (min ≈ 0.57). NOT the mean (~0.76)
    # which would have read "near win" even though both clauses
    # failed (won is False).
    assert g["objective_blocking_ratio"] == pytest.approx(4 / 7, abs=1e-4)
    assert g["objective_progress"] == g["objective_blocking_ratio"]
    # And the mean would have been MISLEADINGLY higher — pin the gap
    # so a future regression to mean-of-leaves trips this test.
    mean_of_leaves = sum(l["ratio"] for l in g["leaves"]) / len(g["leaves"])
    assert mean_of_leaves > g["objective_blocking_ratio"] + 0.1
    assert g["won"] is False


# ---- 3. end-to-end persisted + loadable -----------------------------------


def test_playback_round_trip_has_reasoning_and_goal(tmp_path):
    from openra_bench.playback import Playback
    from openra_bench.playback_view import load_episode

    pb = Playback(tmp_path, "pack:easy", 7)

    class S(_Sig):
        pass

    pb.record_turn(
        1, {"minimap": "..", "units_summary": [], "enemy_summary": []},
        ["Command::Observe"], S(), None,
        goal=turn_goal({"units_killed_gte": 5}, _ctx()),
    )
    pb.write_messages([
        {"role": "system", "content": "s"},
        {"role": "user", "content": "briefing"},
        {"role": "assistant", "content": "go", "reasoning": "scout east",
         "tool_calls": []},
    ])
    pb.finalize({"scenario": "pack:easy", "outcome": "loss", "turns": 1})

    ep = load_episode(pb.dir)
    assert ep["manifest"]["outcome"] == "loss"
    assert ep["turns"][0]["goal"]["won"] is False
    assert ep["turns"][0]["goal"]["leaves"][0]["name"] == "units_killed_gte"
    asst = [m for m in ep["messages"] if m["role"] == "assistant"][0]
    assert asst["reasoning"] == "scout east"
    # turns.jsonl is valid JSONL
    raw = (pb.dir / "turns.jsonl").read_text().strip().splitlines()
    assert json.loads(raw[0])["goal"]["reward_vector"]["territory"] >= 0.0


# ---- 4. leaf table rendering (no percentage scalar) -----------------------


def test_render_leaves_table_shows_current_over_target_not_percent():
    """The viewer must render the per-leaf table verbatim — explicit
    `current/target` plus a satisfied mark — and MUST NOT collapse it
    to a misleading "objective: X%" scalar. Pinning the rendered
    substrings keeps the user-visible defect (the 0.79 "near win"
    that hid two failed clauses) from coming back.
    """
    from openra_bench.playback_view import render_leaves_table

    leaves = [
        {"name": "units_killed_gte", "target": 7, "current": 4,
         "ratio": 4 / 7, "satisfied": False},
        {"name": "within_ticks", "target": 4000, "current": 4203,
         "ratio": 0.0, "satisfied": False},
    ]
    out = render_leaves_table(leaves)
    # the load-bearing substrings every reviewer should see
    assert "4/7" in out
    assert "tick 4203/4000" in out
    # MUST NOT contain a percentage scalar
    assert "%" not in out
    # both clauses failed → x marks present
    assert "units_killed_gte: 4/7 x" in out
    # an empty leaves list renders to the empty string (no goal yet)
    assert render_leaves_table([]) == ""
