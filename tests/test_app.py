"""Tests for the Gradio leaderboard app."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    AGENT_TYPE_COLORS,
    DISPLAY_COLUMNS,
    GAMES_JSONL,
    MAX_SUBMITS_PER_HOUR,
    MIN_GAMES_FOR_LEADERBOARD,
    VALID_OPPONENTS,
    _aggregate_agent_games,
    _check_rate_limit,
    _load_raw_games,
    _rebuild_leaderboard,
    _safe_agent_link,
    _safe_replay_link,
    _sanitize_csv_value,
    _save_raw_game,
    _submit_times,
    add_type_badges,
    build_app,
    filter_leaderboard,
    handle_upload,
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

    def test_valid_json(self, tmp_path):
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
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"
        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            result = handle_api_submit(json.dumps(data))
            assert "OK" in result
            assert "TestBot" in result

    def test_invalid_json(self):
        _submit_times.clear()
        result = handle_api_submit("not json")
        assert "Invalid JSON" in result

    def test_missing_fields(self):
        _submit_times.clear()
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

    def test_agent_url_in_submission(self, tmp_path):
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
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"
        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            result = handle_api_submit(json.dumps(data))
            assert "OK" in result
            assert "DeathBot" in result

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


class TestXssPrevention:
    """Test that user input is HTML-escaped to prevent XSS."""

    def test_javascript_url_blocked(self):
        """javascript: URLs should NOT produce a clickable link."""
        result = _safe_agent_link("Bot", "javascript:alert(1)")
        assert "javascript:" not in result
        assert "Bot" in result

    def test_data_url_blocked(self):
        result = _safe_agent_link("Bot", "data:text/html,<script>alert(1)</script>")
        assert "data:" not in result

    def test_html_in_name_escaped(self):
        result = _safe_agent_link('<script>alert("xss")</script>', "")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_quote_injection_in_url_escaped(self):
        result = _safe_agent_link("Bot", 'https://ok.com" onclick="alert(1)')
        assert 'onclick' not in result or '&quot;' in result

    def test_valid_https_url_works(self):
        result = _safe_agent_link("Bot", "https://github.com/user/repo")
        assert '<a href="https://github.com/user/repo"' in result
        assert 'rel="noopener"' in result

    def test_replay_link_sanitized(self):
        result = _safe_replay_link('"><script>alert(1)</script>.orarep')
        assert "<script>" not in result

    def test_replay_path_traversal_stripped(self):
        """Path traversal characters (/) are stripped from replay filenames."""
        result = _safe_replay_link("replay/../../../etc/passwd")
        # The href after /replays/ should have no slashes (traversal stripped)
        href_part = result.split('href="')[1].split('"')[0]
        filename = href_part.replace("/replays/", "")
        assert "/" not in filename


class TestInputValidation:
    """Test stricter input validation."""

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

    def test_string_ticks_rejected(self):
        data = self._valid_data()
        data["ticks"] = "not a number"
        valid, err = validate_submission(data)
        assert not valid
        assert "must be a number" in err

    def test_dict_kills_rejected(self):
        data = self._valid_data()
        data["kills_cost"] = {"nested": True}
        valid, err = validate_submission(data)
        assert not valid

    def test_long_agent_name_rejected(self):
        data = self._valid_data()
        data["agent_name"] = "A" * 101
        valid, err = validate_submission(data)
        assert not valid
        assert "100 characters" in err

    def test_javascript_agent_url_rejected(self):
        data = self._valid_data()
        data["agent_url"] = "javascript:alert(1)"
        valid, err = validate_submission(data)
        assert not valid
        assert "HTTP(S)" in err

    def test_valid_agent_url_accepted(self):
        data = self._valid_data()
        data["agent_url"] = "https://github.com/user/repo"
        valid, _ = validate_submission(data)
        assert valid

    def test_empty_agent_url_accepted(self):
        data = self._valid_data()
        data["agent_url"] = ""
        valid, _ = validate_submission(data)
        assert valid

    def test_long_agent_url_rejected(self):
        data = self._valid_data()
        data["agent_url"] = "https://example.com/" + "a" * 500
        valid, err = validate_submission(data)
        assert not valid
        assert "500 characters" in err


class TestCsvSanitization:
    """Test CSV injection prevention."""

    def test_formula_trigger_stripped(self):
        assert _sanitize_csv_value("=cmd|'/c calc'!A0") == "cmd|'/c calc'!A0"

    def test_plus_trigger_stripped(self):
        assert _sanitize_csv_value("+cmd") == "cmd"

    def test_at_trigger_stripped(self):
        assert _sanitize_csv_value("@SUM(A1)") == "SUM(A1)"

    def test_newlines_replaced(self):
        assert _sanitize_csv_value("line1\nline2\rline3") == "line1 line2 line3"

    def test_normal_string_unchanged(self):
        assert _sanitize_csv_value("DeathBot-9000") == "DeathBot-9000"

    def test_numbers_unchanged(self):
        assert _sanitize_csv_value(42) == 42
        assert _sanitize_csv_value(3.14) == 3.14


class TestRateLimiting:
    """Test rate limiting on submissions."""

    def test_rate_limit_allows_normal_usage(self):
        _submit_times.clear()
        allowed, _ = _check_rate_limit("test_normal")
        assert allowed

    def test_rate_limit_blocks_after_max(self):
        _submit_times.clear()
        key = "test_flood"
        for _ in range(MAX_SUBMITS_PER_HOUR):
            allowed, _ = _check_rate_limit(key)
            assert allowed
        allowed, err = _check_rate_limit(key)
        assert not allowed
        assert "Rate limit" in err

    def test_rate_limit_resets_after_expiry(self):
        import time as _time
        _submit_times.clear()
        key = "test_expiry"
        # Fill with old timestamps
        _submit_times[key] = [_time.time() - 3601] * MAX_SUBMITS_PER_HOUR
        allowed, _ = _check_rate_limit(key)
        assert allowed


class TestSearchSafety:
    """Test that malformed regex doesn't crash the search."""

    def test_invalid_regex_falls_back(self):
        """An invalid regex pattern should not raise an exception."""
        df = filter_leaderboard("[invalid(regex", [], "All")
        assert isinstance(df, pd.DataFrame)


# ── New aggregation tests ────────────────────────────────────────────────────


class TestSaveRawGame:
    """Test raw game storage."""

    def test_appends_to_jsonl(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            _save_raw_game({"agent_name": "Bot1", "result": "win"})
            _save_raw_game({"agent_name": "Bot2", "result": "lose"})

        lines = games_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["agent_name"] == "Bot1"
        assert json.loads(lines[1])["agent_name"] == "Bot2"

    def test_also_writes_results_jsonl(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            _save_raw_game({"agent_name": "Bot1", "result": "win"})

        jsonl_path = tmp_path / "results.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["agent_name"] == "Bot1"


class TestLoadRawGames:
    """Test raw game loading."""

    def test_returns_empty_when_no_file(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        with patch("app.GAMES_JSONL", games_path):
            assert _load_raw_games() == []

    def test_loads_multiple_games(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        games_path.write_text(
            '{"agent_name": "A"}\n'
            '{"agent_name": "B"}\n'
        )
        with patch("app.GAMES_JSONL", games_path):
            games = _load_raw_games()
        assert len(games) == 2
        assert games[0]["agent_name"] == "A"

    def test_skips_invalid_json_lines(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        games_path.write_text(
            '{"agent_name": "A"}\n'
            'not valid json\n'
            '{"agent_name": "B"}\n'
        )
        with patch("app.GAMES_JSONL", games_path):
            games = _load_raw_games()
        assert len(games) == 2

    def test_skips_blank_lines(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        games_path.write_text(
            '{"agent_name": "A"}\n'
            '\n'
            '  \n'
            '{"agent_name": "B"}\n'
        )
        with patch("app.GAMES_JSONL", games_path):
            games = _load_raw_games()
        assert len(games) == 2


class TestAggregation:
    """Test game aggregation logic."""

    def _make_game(self, agent="Bot", opponent="Normal", win=True, kills=1000, deaths=500, assets=5000, ticks=2000):
        return {
            "agent_name": agent,
            "agent_type": "RL",
            "opponent": opponent,
            "result": "win" if win else "lose",
            "win": win,
            "kills_cost": kills,
            "deaths_cost": deaths,
            "assets_value": assets,
            "ticks": ticks,
            "timestamp": "2026-03-02",
        }

    def test_below_threshold_returns_none(self):
        games = [self._make_game() for _ in range(MIN_GAMES_FOR_LEADERBOARD - 1)]
        count, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert count == MIN_GAMES_FOR_LEADERBOARD - 1
        assert agg is None

    def test_at_threshold_returns_row(self):
        games = [self._make_game() for _ in range(MIN_GAMES_FOR_LEADERBOARD)]
        count, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert count == MIN_GAMES_FOR_LEADERBOARD
        assert agg is not None
        assert agg["games"] == MIN_GAMES_FOR_LEADERBOARD
        assert agg["win_rate"] == 100.0

    def test_applies_difficulty_multiplier(self):
        games_normal = [self._make_game(opponent="Normal") for _ in range(5)]
        games_hard = [self._make_game(opponent="Hard") for _ in range(5)]
        _, agg_normal = _aggregate_agent_games("Bot", "RL", "Normal", games_normal)
        _, agg_hard = _aggregate_agent_games("Bot", "RL", "Hard", games_hard)
        assert agg_hard["score"] > agg_normal["score"]

    def test_filters_by_agent_and_opponent(self):
        games = (
            [self._make_game(agent="Bot1", opponent="Normal") for _ in range(5)]
            + [self._make_game(agent="Bot2", opponent="Normal") for _ in range(3)]
            + [self._make_game(agent="Bot1", opponent="Hard") for _ in range(2)]
        )
        count1, agg1 = _aggregate_agent_games("Bot1", "RL", "Normal", games)
        count2, _ = _aggregate_agent_games("Bot2", "RL", "Normal", games)
        count3, _ = _aggregate_agent_games("Bot1", "RL", "Hard", games)
        assert count1 == 5
        assert agg1 is not None
        assert count2 == 3  # Below threshold
        assert count3 == 2  # Below threshold

    def test_mixed_results_aggregated(self):
        games = [
            self._make_game(win=True, kills=5000, deaths=1000),
            self._make_game(win=True, kills=3000, deaths=2000),
            self._make_game(win=False, kills=1000, deaths=4000),
            self._make_game(win=True, kills=4000, deaths=1500),
            self._make_game(win=False, kills=2000, deaths=3000),
        ]
        count, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert count == 5
        assert agg["win_rate"] == 60.0
        assert agg["avg_kills"] == 3000  # (5000+3000+1000+4000+2000)/5
        assert agg["games"] == 5

    def test_kd_ratio_computed(self):
        games = [self._make_game(kills=2000, deaths=1000) for _ in range(5)]
        _, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert agg["kd_ratio"] == 2.0

    def test_zero_deaths_kd_ratio(self):
        games = [self._make_game(kills=1000, deaths=0) for _ in range(5)]
        _, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert agg["kd_ratio"] == 5000.0  # 5000 total kills / max(0, 1)

    def test_aggregation_includes_metadata(self):
        games = [self._make_game() for _ in range(5)]
        games[-1]["agent_url"] = "https://github.com/user/bot"
        games[-1]["replay_url"] = "replay-bot.orarep"
        _, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert agg["agent_url"] == "https://github.com/user/bot"
        assert agg["replay_url"] == "replay-bot.orarep"
        assert agg["difficulty"] == "Normal"

    def test_filters_by_agent_type(self):
        """Different agent_types for the same name should not mix."""
        games = (
            [self._make_game(agent="Bot") for _ in range(5)]  # agent_type="RL"
        )
        # Query with wrong agent_type
        count, agg = _aggregate_agent_games("Bot", "LLM", "Normal", games)
        assert count == 0
        assert agg is None

    def test_win_field_fallback_to_result(self):
        """When 'win' field is missing, fall back to result=='win'."""
        games = []
        for _ in range(5):
            g = self._make_game()
            del g["win"]
            g["result"] = "win"
            games.append(g)
        _, agg = _aggregate_agent_games("Bot", "RL", "Normal", games)
        assert agg["win_rate"] == 100.0


class TestRebuildLeaderboard:
    """Test leaderboard rebuild from raw games."""

    def test_filters_below_threshold(self, tmp_path):
        data_path = tmp_path / "results.csv"
        games_path = tmp_path / "games.jsonl"

        with patch("app.DATA_PATH", data_path), \
             patch("app.GAMES_JSONL", games_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            # Save 3 games for Bot1 (below threshold)
            for _ in range(3):
                _save_raw_game({
                    "agent_name": "Bot1", "agent_type": "RL", "opponent": "Normal",
                    "result": "win", "win": True, "kills_cost": 1000,
                    "deaths_cost": 500, "assets_value": 5000, "ticks": 2000,
                    "timestamp": "2026-03-02",
                })
            # Save 5 games for Bot2 (at threshold)
            for _ in range(5):
                _save_raw_game({
                    "agent_name": "Bot2", "agent_type": "RL", "opponent": "Normal",
                    "result": "win", "win": True, "kills_cost": 2000,
                    "deaths_cost": 1000, "assets_value": 8000, "ticks": 1500,
                    "timestamp": "2026-03-02",
                })

            _rebuild_leaderboard()

        df = pd.read_csv(data_path)
        assert len(df) == 1  # Only Bot2 appears
        assert df.iloc[0]["agent_name"] == "Bot2"
        assert df.iloc[0]["games"] == 5

    def test_rebuild_sorts_by_score(self, tmp_path):
        data_path = tmp_path / "results.csv"
        games_path = tmp_path / "games.jsonl"

        with patch("app.DATA_PATH", data_path), \
             patch("app.GAMES_JSONL", games_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            # Bot with worse stats
            for _ in range(5):
                _save_raw_game({
                    "agent_name": "WeakBot", "agent_type": "RL", "opponent": "Normal",
                    "result": "lose", "win": False, "kills_cost": 100,
                    "deaths_cost": 5000, "assets_value": 1000, "ticks": 5000,
                    "timestamp": "2026-03-02",
                })
            # Bot with better stats
            for _ in range(5):
                _save_raw_game({
                    "agent_name": "StrongBot", "agent_type": "RL", "opponent": "Normal",
                    "result": "win", "win": True, "kills_cost": 5000,
                    "deaths_cost": 500, "assets_value": 20000, "ticks": 1000,
                    "timestamp": "2026-03-02",
                })

            _rebuild_leaderboard()

        df = pd.read_csv(data_path)
        assert len(df) == 2
        assert df.iloc[0]["agent_name"] == "StrongBot"
        assert df.iloc[1]["agent_name"] == "WeakBot"

    def test_rebuild_no_games_preserves_csv(self, tmp_path):
        data_path = tmp_path / "results.csv"
        games_path = tmp_path / "games.jsonl"

        # Create an existing CSV
        data_path.write_text("agent_name,score\nOldBot,50\n")

        with patch("app.DATA_PATH", data_path), \
             patch("app.GAMES_JSONL", games_path):
            _rebuild_leaderboard()

        # File should be unchanged (no games.jsonl = no rebuild)
        assert "OldBot" in data_path.read_text()

    def test_rebuild_no_qualifying_agents_preserves_csv(self, tmp_path):
        data_path = tmp_path / "results.csv"
        games_path = tmp_path / "games.jsonl"

        data_path.write_text("agent_name,score\nOldBot,50\n")

        with patch("app.DATA_PATH", data_path), \
             patch("app.GAMES_JSONL", games_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            # Only 2 games — no one qualifies
            for _ in range(2):
                _save_raw_game({
                    "agent_name": "Bot1", "agent_type": "RL", "opponent": "Normal",
                    "result": "win", "win": True, "kills_cost": 1000,
                    "deaths_cost": 500, "assets_value": 5000, "ticks": 2000,
                    "timestamp": "2026-03-02",
                })

            _rebuild_leaderboard()

        # CSV still has old data
        assert "OldBot" in data_path.read_text()

    def test_rebuild_includes_difficulty_column(self, tmp_path):
        data_path = tmp_path / "results.csv"
        games_path = tmp_path / "games.jsonl"

        with patch("app.DATA_PATH", data_path), \
             patch("app.GAMES_JSONL", games_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            for _ in range(5):
                _save_raw_game({
                    "agent_name": "Bot1", "agent_type": "RL", "opponent": "Hard",
                    "result": "win", "win": True, "kills_cost": 1000,
                    "deaths_cost": 500, "assets_value": 5000, "ticks": 2000,
                    "timestamp": "2026-03-02",
                })
            _rebuild_leaderboard()

        df = pd.read_csv(data_path)
        assert "difficulty" in df.columns
        assert df.iloc[0]["difficulty"] == "Hard"


class TestFriendlyMessages:
    """Test submission response messages."""

    def test_below_threshold_message(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"

        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            data = {
                "agent_name": "TestBot", "agent_type": "RL", "opponent": "Normal",
                "result": "win", "ticks": 2000, "kills_cost": 1000,
                "deaths_cost": 500, "assets_value": 5000,
            }
            result = handle_api_submit(json.dumps(data))

        assert "1/5" in result
        assert "Play 4 more" in result
        assert "leaderboard" in result

    def test_at_threshold_message(self, tmp_path):
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"

        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            data = {
                "agent_name": "TestBot", "agent_type": "RL", "opponent": "Normal",
                "result": "win", "ticks": 2000, "kills_cost": 1000,
                "deaths_cost": 500, "assets_value": 5000,
            }
            # Submit 5 games
            for _ in range(4):
                handle_api_submit(json.dumps(data))
            result = handle_api_submit(json.dumps(data))

        assert "updated" in result
        assert "5 games" in result
        assert "score" in result

    def test_singular_remaining_game(self, tmp_path):
        """When only 1 game remains, message should say 'game' not 'games'."""
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"

        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            data = {
                "agent_name": "TestBot", "agent_type": "RL", "opponent": "Normal",
                "result": "win", "ticks": 2000, "kills_cost": 1000,
                "deaths_cost": 500, "assets_value": 5000,
            }
            # Submit 4 games
            for _ in range(3):
                handle_api_submit(json.dumps(data))
            result = handle_api_submit(json.dumps(data))

        assert "4/5" in result
        assert "Play 1 more game " in result  # No 's' — singular

    def test_progress_increments(self, tmp_path):
        """Each submission should increment the game count."""
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"

        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            data = {
                "agent_name": "TestBot", "agent_type": "RL", "opponent": "Normal",
                "result": "win", "ticks": 2000, "kills_cost": 1000,
                "deaths_cost": 500, "assets_value": 5000,
            }
            r1 = handle_api_submit(json.dumps(data))
            r2 = handle_api_submit(json.dumps(data))
            r3 = handle_api_submit(json.dumps(data))

        assert "1/5" in r1
        assert "2/5" in r2
        assert "3/5" in r3


class TestHandleUploadAggregation:
    """Test the UI upload handler with aggregation."""

    def _valid_data(self):
        return {
            "agent_name": "UploadBot",
            "agent_type": "LLM",
            "opponent": "Easy",
            "result": "win",
            "ticks": 3000,
            "kills_cost": 2000,
            "deaths_cost": 800,
            "assets_value": 7000,
        }

    def test_upload_below_threshold_message(self, tmp_path):
        """Upload handler should show progress message below threshold."""
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"

        # Create a JSON file for upload
        json_path = tmp_path / "upload.json"
        json_path.write_text(json.dumps(self._valid_data()))

        class FakeFile:
            def __init__(self, p):
                self.name = str(p)

        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            msg, df = handle_upload(FakeFile(json_path), None)

        assert "1/5" in msg
        assert "UploadBot" in msg
        assert "leaderboard" in msg

    def test_upload_at_threshold_message(self, tmp_path):
        """Upload handler should show score when threshold reached."""
        games_path = tmp_path / "games.jsonl"
        data_path = tmp_path / "results.csv"
        data = self._valid_data()

        # Pre-seed 4 games
        for _ in range(4):
            with open(games_path, "a") as f:
                f.write(json.dumps(data) + "\n")

        json_path = tmp_path / "upload.json"
        json_path.write_text(json.dumps(data))

        class FakeFile:
            def __init__(self, p):
                self.name = str(p)

        _submit_times.clear()
        with patch("app.GAMES_JSONL", games_path), \
             patch("app.DATA_PATH", data_path), \
             patch("app.SUBMISSIONS_DIR", tmp_path):
            msg, df = handle_upload(FakeFile(json_path), None)

        assert "updated" in msg
        assert "5 games" in msg
        assert "score" in msg
