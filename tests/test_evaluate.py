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
        assert len(RESULTS_COLUMNS) == 13


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
