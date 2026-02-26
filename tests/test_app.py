"""Tests for the Gradio leaderboard app."""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    AGENT_TYPE_COLORS,
    DISPLAY_COLUMNS,
    VALID_OPPONENTS,
    add_type_badges,
    build_app,
    filter_leaderboard,
    handle_api_submit,
    handle_api_submit_with_replay,
    load_data,
    validate_submission,
)


class TestLoadData:
    """Test data loading."""

    def test_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_display_columns(self):
        df = load_data()
        for col in DISPLAY_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_rank_column(self):
        df = load_data()
        if len(df) > 0:
            assert df["Rank"].iloc[0] == 1

    def test_sorted_by_score_descending(self):
        df = load_data()
        if len(df) > 1:
            scores = df["Score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_handles_missing_file(self):
        with patch("app.DATA_PATH", Path("/nonexistent/data.csv")):
            df = load_data()
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0


class TestBadges:
    """Test type badge rendering."""

    def test_scripted_badge_has_gold(self):
        df = pd.DataFrame({"Type": ["Scripted"]})
        result = add_type_badges(df)
        assert "#ffcd75" in result["Type"].iloc[0]

    def test_llm_badge_has_blue(self):
        df = pd.DataFrame({"Type": ["LLM"]})
        result = add_type_badges(df)
        assert "#7497db" in result["Type"].iloc[0]

    def test_rl_badge_has_gray(self):
        df = pd.DataFrame({"Type": ["RL"]})
        result = add_type_badges(df)
        assert "#75809c" in result["Type"].iloc[0]

    def test_all_types_have_colors(self):
        for t in ["Scripted", "LLM", "RL"]:
            assert t in AGENT_TYPE_COLORS


class TestFilter:
    """Test leaderboard filtering."""

    def test_returns_dataframe(self):
        df = filter_leaderboard("", [], "All")
        assert isinstance(df, pd.DataFrame)

    def test_search_filters_by_name(self):
        df = filter_leaderboard("qwen", [], "All")
        if len(df) > 0:
            assert all("qwen" in str(row).lower() for row in df["Agent"])

    def test_opponent_filter(self):
        df = filter_leaderboard("", [], "Beginner")
        if len(df) > 0:
            assert all(df["Opponent"] == "Beginner")

    def test_opponent_filter_hard(self):
        df = filter_leaderboard("", [], "Hard")
        # May be empty if no Hard entries exist
        assert isinstance(df, pd.DataFrame)


class TestBuildApp:
    """Test app construction."""

    def test_builds_without_error(self):
        app = build_app()
        assert app is not None


class TestValidateSubmission:
    """Test submission validation."""

    def _valid_data(self):
        return {
            "agent_name": "TestBot",
            "agent_type": "LLM",
            "opponent": "Beginner",
            "result": "loss",
            "ticks": 27000,
            "kills_cost": 1000,
            "deaths_cost": 2900,
            "assets_value": 9050,
        }

    def test_valid_submission(self):
        valid, err = validate_submission(self._valid_data())
        assert valid
        assert err == ""

    def test_missing_field(self):
        data = {"agent_name": "Bot"}
        valid, err = validate_submission(data)
        assert not valid
        assert "Missing required field" in err

    def test_invalid_opponent(self):
        data = self._valid_data()
        data["opponent"] = "Brutal"
        valid, err = validate_submission(data)
        assert not valid
        assert "Invalid opponent" in err

    def test_invalid_agent_type(self):
        data = self._valid_data()
        data["agent_type"] = "MCTS"
        valid, err = validate_submission(data)
        assert not valid
        assert "Invalid agent_type" in err

    def test_all_opponents_accepted(self):
        for opp in VALID_OPPONENTS:
            data = self._valid_data()
            data["opponent"] = opp
            valid, _ = validate_submission(data)
            assert valid, f"Opponent '{opp}' should be valid"

    def test_all_agent_types_accepted(self):
        for at in ["Scripted", "LLM", "RL"]:
            data = self._valid_data()
            data["agent_type"] = at
            valid, _ = validate_submission(data)
            assert valid, f"Agent type '{at}' should be valid"


class TestApiSubmit:
    """Test API submission handler."""

    def test_valid_json(self):
        import json
        data = {
            "agent_name": "TestBot",
            "agent_type": "LLM",
            "opponent": "Easy",
            "result": "win",
            "win": True,
            "ticks": 5000,
            "kills_cost": 3000,
            "deaths_cost": 1000,
            "assets_value": 8000,
        }
        # Use a temp CSV to avoid polluting real data
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = Path(f.name)
        with patch("app.DATA_PATH", temp_path):
            result = handle_api_submit(json.dumps(data))
            assert "OK" in result
            assert "TestBot" in result
        temp_path.unlink(missing_ok=True)

    def test_invalid_json(self):
        result = handle_api_submit("not json")
        assert "Invalid JSON" in result

    def test_missing_fields(self):
        import json
        result = handle_api_submit(json.dumps({"agent_name": "Bot"}))
        assert "Validation error" in result


class TestDisplayColumns:
    """Test display column configuration."""

    def test_replay_in_display_columns(self):
        assert "Replay" in DISPLAY_COLUMNS

    def test_display_columns_count(self):
        assert len(DISPLAY_COLUMNS) == 14


class TestAgentUrl:
    """Test agent URL hyperlink rendering."""

    def test_agent_url_in_submission(self):
        import json
        import tempfile
        data = {
            "agent_name": "DeathBot",
            "agent_type": "RL",
            "agent_url": "https://github.com/user/deathbot",
            "opponent": "Normal",
            "result": "win",
            "ticks": 5000,
            "kills_cost": 3000,
            "deaths_cost": 1000,
            "assets_value": 8000,
        }
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = Path(f.name)
        with patch("app.DATA_PATH", temp_path):
            result = handle_api_submit(json.dumps(data))
            assert "OK" in result
            assert "DeathBot" in result
        temp_path.unlink(missing_ok=True)

    def test_agent_url_renders_link(self):
        """When agent_url is set, Agent column should be a hyperlink."""
        import tempfile
        csv_content = (
            "agent_name,agent_type,opponent,games,win_rate,score,"
            "avg_kills,avg_deaths,kd_ratio,avg_economy,avg_game_length,"
            "timestamp,replay_url,agent_url\n"
            "DeathBot,RL,Normal,10,50.0,60.0,"
            "2000,1500,1.33,9000,15000,"
            "2026-02-26,,https://github.com/user/deathbot\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        with patch("app.DATA_PATH", temp_path):
            df = load_data()
            assert '<a href="https://github.com/user/deathbot"' in df["Agent"].iloc[0]
        temp_path.unlink(missing_ok=True)

    def test_no_url_renders_plain_name(self):
        """When agent_url is empty, Agent column is plain text."""
        import tempfile
        csv_content = (
            "agent_name,agent_type,opponent,games,win_rate,score,"
            "avg_kills,avg_deaths,kd_ratio,avg_economy,avg_game_length,"
            "timestamp,replay_url,agent_url\n"
            "PlainBot,LLM,Easy,5,20.0,30.0,"
            "1000,2000,0.5,5000,10000,"
            "2026-02-26,,\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        with patch("app.DATA_PATH", temp_path):
            df = load_data()
            assert df["Agent"].iloc[0] == "PlainBot"
        temp_path.unlink(missing_ok=True)


class TestReplayColumn:
    """Test replay download link rendering."""

    def test_replay_link_rendered(self):
        """Replay column shows download link when replay_url is set."""
        import tempfile
        csv_content = (
            "agent_name,agent_type,opponent,games,win_rate,score,"
            "avg_kills,avg_deaths,kd_ratio,avg_economy,avg_game_length,"
            "timestamp,replay_url,agent_url\n"
            "TestBot,LLM,Easy,1,0.0,18.0,"
            "1000,2000,0.5,5000,10000,"
            "2026-02-26,replay-test-123.orarep,\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        with patch("app.DATA_PATH", temp_path):
            df = load_data()
            assert "/replays/replay-test-123.orarep" in df["Replay"].iloc[0]
            assert "download" in df["Replay"].iloc[0]
        temp_path.unlink(missing_ok=True)

    def test_empty_replay_no_link(self):
        """Replay column is empty when no replay_url."""
        df = load_data()
        if len(df) > 0:
            # The default test data has no replay
            replay_val = df["Replay"].iloc[0]
            assert replay_val == "" or not str(replay_val).strip()
