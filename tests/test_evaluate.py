"""Tests for the evaluation harness."""

import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate import (
    RESULTS_COLUMNS,
    append_results,
    get_agent_fn,
    parse_args,
)


class TestParseArgs:
    """Test argument parsing."""

    def test_minimal_args(self):
        with patch("sys.argv", ["evaluate.py", "--agent-name", "TestBot"]):
            args = parse_args()
            assert args.agent_name == "TestBot"
            assert args.agent == "scripted"
            assert args.agent_type == "Scripted"
            assert args.opponent == "Normal"
            assert args.games == 10

    def test_all_args(self):
        with patch("sys.argv", [
            "evaluate.py",
            "--agent", "llm",
            "--agent-name", "MyLLM",
            "--agent-type", "LLM",
            "--opponent", "Hard",
            "--games", "5",
            "--server", "http://example.com:8000",
            "--max-steps", "3000",
            "--dry-run",
        ]):
            args = parse_args()
            assert args.agent == "llm"
            assert args.agent_name == "MyLLM"
            assert args.agent_type == "LLM"
            assert args.opponent == "Hard"
            assert args.games == 5
            assert args.server == "http://example.com:8000"
            assert args.max_steps == 3000
            assert args.dry_run is True

    def test_auto_detect_agent_type(self):
        for agent, expected_type in [
            ("scripted", "Scripted"),
            ("llm", "LLM"),
            ("mcp", "Scripted"),
            ("custom", "RL"),
        ]:
            with patch("sys.argv", ["evaluate.py", "--agent", agent, "--agent-name", "T"]):
                args = parse_args()
                assert args.agent_type == expected_type, f"{agent} -> {expected_type}"

    def test_explicit_type_overrides_auto(self):
        with patch("sys.argv", [
            "evaluate.py", "--agent", "scripted",
            "--agent-name", "T", "--agent-type", "RL",
        ]):
            args = parse_args()
            assert args.agent_type == "RL"

    def test_beginner_opponent_accepted(self):
        with patch("sys.argv", [
            "evaluate.py", "--agent-name", "T", "--opponent", "Beginner",
        ]):
            args = parse_args()
            assert args.opponent == "Beginner"

    def test_medium_opponent_accepted(self):
        with patch("sys.argv", [
            "evaluate.py", "--agent-name", "T", "--opponent", "Medium",
        ]):
            args = parse_args()
            assert args.opponent == "Medium"


class TestGetAgentFn:
    """Test agent factory."""

    def test_scripted_returns_callable(self):
        fn = get_agent_fn("scripted")
        assert callable(fn)

    def test_llm_returns_callable(self):
        fn = get_agent_fn("llm")
        assert callable(fn)


class TestAppendResults:
    """Test CSV output."""

    def test_creates_new_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        path.unlink()  # ensure it doesn't exist
        results = {col: "" for col in RESULTS_COLUMNS}
        results["agent_name"] = "TestBot"
        results["games"] = 5
        results["score"] = 85.0

        append_results(results, path)

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["agent_name"] == "TestBot"

        path.unlink()

    def test_appends_to_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = Path(f.name)

        # Write first result
        results1 = {col: "" for col in RESULTS_COLUMNS}
        results1["agent_name"] = "Bot1"
        append_results(results1, path)

        # Write second result
        results2 = {col: "" for col in RESULTS_COLUMNS}
        results2["agent_name"] = "Bot2"
        append_results(results2, path)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["agent_name"] == "Bot1"
            assert rows[1]["agent_name"] == "Bot2"

        path.unlink()

    def test_columns_match_expected(self):
        assert "agent_name" in RESULTS_COLUMNS
        assert "score" in RESULTS_COLUMNS
        assert "win_rate" in RESULTS_COLUMNS
        assert "replay_url" in RESULTS_COLUMNS
        assert len(RESULTS_COLUMNS) == 14


class TestScoringUsesUtil:
    """Verify scoring uses the single source of truth from openra-rl-util."""

    def test_rubrics_re_exports_util(self):
        """rubrics.py should re-export from openra_rl_util."""
        from rubrics import compute_composite_score_from_games
        from openra_rl_util.rubrics import (
            compute_composite_score_from_games as util_fn,
        )
        assert compute_composite_score_from_games is util_fn

    def test_evaluate_uses_util_scoring(self):
        """evaluate.py should not have its own compute_composite_score."""
        import evaluate
        assert not hasattr(evaluate, "compute_composite_score"), \
            "evaluate.py should use compute_composite_score_from_games from Util"

    def test_compute_game_metrics_re_exported(self):
        from rubrics import compute_game_metrics
        from openra_rl_util.rubrics import compute_game_metrics as util_fn
        assert compute_game_metrics is util_fn


class TestMinGames:
    """Minimum game count validation."""

    def test_evaluate_runner_rejects_few_games(self):
        """run_evaluation should reject num_games < 5."""
        import pytest
        from evaluate_runner import run_evaluation
        with pytest.raises(ValueError, match="Minimum 5 games"):
            import asyncio
            asyncio.run(run_evaluation("test", "Normal", 3, "http://fake"))


class TestDifficultyMultiplier:
    """Difficulty scaling for benchmark scores."""

    def test_hard_scores_higher(self):
        from evaluate_runner import DIFFICULTY_MULTIPLIER
        assert DIFFICULTY_MULTIPLIER["Hard"] > DIFFICULTY_MULTIPLIER["Normal"]

    def test_beginner_scores_lower(self):
        from evaluate_runner import DIFFICULTY_MULTIPLIER
        assert DIFFICULTY_MULTIPLIER["Beginner"] < DIFFICULTY_MULTIPLIER["Normal"]

    def test_normal_is_baseline(self):
        from evaluate_runner import DIFFICULTY_MULTIPLIER
        assert DIFFICULTY_MULTIPLIER["Normal"] == 1.0


class TestCompositeScoreUpdate:
    """Updated composite score formula."""

    def test_no_combat_military_is_zero(self):
        from evaluate_runner import compute_composite_score
        games = [
            {"win": True, "kills_cost": 0, "deaths_cost": 0, "assets_value": 10000, "ticks": 2000},
        ]
        score = compute_composite_score(games)
        # 50%*1.0 + 20%*0.0 + 20%*0.5 + 10%*(1/(1+2/3))
        # = 50 + 0 + 10 + 6.0 = 66.0
        assert abs(score - 66.0) < 1.0

    def test_speed_bonus_for_fast_games(self):
        from evaluate_runner import compute_composite_score
        fast = [{"win": True, "kills_cost": 5000, "deaths_cost": 1000, "assets_value": 10000, "ticks": 500}]
        slow = [{"win": True, "kills_cost": 5000, "deaths_cost": 1000, "assets_value": 10000, "ticks": 10000}]
        assert compute_composite_score(fast) > compute_composite_score(slow)
