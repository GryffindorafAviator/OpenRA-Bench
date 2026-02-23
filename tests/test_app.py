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
    add_type_badges,
    build_app,
    filter_leaderboard,
    load_data,
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
        df = filter_leaderboard("ScriptedBot", [], "All")
        # If there are results, they should contain "ScriptedBot"
        if len(df) > 0:
            # Badges are in the Type column, not Agent
            assert all("ScriptedBot" in str(row) for row in df["Agent"])

    def test_opponent_filter(self):
        df = filter_leaderboard("", [], "Hard")
        if len(df) > 0:
            assert all(df["Opponent"] == "Hard")


class TestBuildApp:
    """Test app construction."""

    def test_builds_without_error(self):
        app = build_app()
        assert app is not None
