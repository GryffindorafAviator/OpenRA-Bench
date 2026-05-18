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
    assert "objective_progress" in g and 0.0 <= g["objective_progress"] <= 1.0
    assert g["won"] is False  # 3 < 5


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
