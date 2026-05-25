"""Tests for the Gradio leaderboard app.

The legacy Play / Playlist / Submit tabs and the HuggingFace submission
pipeline were removed (human play now lives in ``site/index.html``;
HF uploads are no longer needed). These tests only cover the surviving
surface: data loading, badges, filtering, the Scenarios tab, XSS-safe
link rendering, and ``build_app()`` smoke.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    AGENT_TYPE_COLORS,
    DISPLAY_COLUMNS,
    _safe_agent_link,
    _safe_replay_link,
    _scenarios_catalog_df,
    _scenarios_detail_md,
    _scenarios_filter,
    _verified_badge,
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


class TestVerifiedBadge:
    def test_verified_true(self):
        assert "Verified" in _verified_badge(True)
        assert "#4caf50" in _verified_badge(True)

    def test_verified_false(self):
        assert "Unverified" in _verified_badge(False)
        assert "#ff9800" in _verified_badge(False)

    def test_verified_string_true(self):
        assert "Verified" in _verified_badge("true")
        assert "Unverified" not in _verified_badge("true")

    def test_verified_string_false(self):
        assert "Unverified" in _verified_badge("false")


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
        assert isinstance(df, pd.DataFrame)


class TestSearchSafety:
    """Test that malformed regex doesn't crash the search."""

    def test_invalid_regex_falls_back(self):
        df = filter_leaderboard("[invalid(regex", [], "All")
        assert isinstance(df, pd.DataFrame)


class TestBuildApp:
    """Test app construction."""

    def test_builds_without_error(self):
        app = build_app()
        assert app is not None


class TestDisplayColumns:
    """Test display column configuration."""

    def test_replay_in_display_columns(self):
        assert "Replay" in DISPLAY_COLUMNS

    def test_display_columns_count(self):
        assert len(DISPLAY_COLUMNS) == 15


class TestAgentUrl:
    """Test agent URL hyperlink rendering."""

    def test_agent_url_renders_link(self):
        """When agent_url is set, Agent column should be a hyperlink."""
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
        df = load_data()
        if len(df) > 0:
            replay_val = df["Replay"].iloc[0]
            assert replay_val == "" or not str(replay_val).strip()


class TestXssPrevention:
    """Test that user input is HTML-escaped to prevent XSS."""

    def test_javascript_url_blocked(self):
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
        result = _safe_replay_link("replay/../../../etc/passwd")
        href_part = result.split('href="')[1].split('"')[0]
        filename = href_part.replace("/replays/", "")
        assert "/" not in filename


# ── Scenarios tab (interactive catalog) ──────────────────────────────


class TestScenariosTab:
    """The Scenarios tab — interactive catalog of all active packs."""

    def test_catalog_df_returns_dataframe(self):
        df = _scenarios_catalog_df()
        assert isinstance(df, pd.DataFrame)

    def test_catalog_df_has_expected_columns(self):
        df = _scenarios_catalog_df()
        for col in ("ID", "Title", "Capability", "Map",
                     "Real-World Meaning", "Robotics Analogue"):
            assert col in df.columns, f"Missing column: {col}"

    def test_catalog_df_has_active_packs(self):
        df = _scenarios_catalog_df()
        assert len(df) > 10, "Expected many active packs"

    def test_catalog_df_only_active_packs(self):
        df = _scenarios_catalog_df()
        assert len(df) > 0
        caps = set(df["Capability"].unique())
        assert caps <= {"perception", "reasoning", "action", "adversarial"}

    def test_filter_by_capability(self):
        df = _scenarios_filter("", ["perception"])
        assert len(df) > 0
        assert all(df["Capability"] == "perception")

    def test_filter_by_search(self):
        df = _scenarios_filter("combat", [
            "perception", "reasoning", "action", "adversarial"
        ])
        assert len(df) > 0
        for _, row in df.iterrows():
            matched = (
                "combat" in row["ID"].lower()
                or "combat" in row["Title"].lower()
                or "combat" in row["Real-World Meaning"].lower()
            )
            assert matched, f"Row {row['ID']} doesn't match 'combat'"

    def test_filter_empty_capabilities_returns_empty(self):
        df = _scenarios_filter("", [])
        assert len(df) == 0

    def test_filter_no_match_returns_empty(self):
        df = _scenarios_filter("zzz_nonexistent_pack_zzz", [
            "perception", "reasoning", "action", "adversarial"
        ])
        assert len(df) == 0

    def test_detail_md_no_selection(self):
        md = _scenarios_detail_md("")
        assert "select" in md.lower() or "Select" in md

    def test_detail_md_nonexistent_pack(self):
        md = _scenarios_detail_md("zzz-no-such-pack-zzz")
        assert "not found" in md.lower()

    def test_detail_md_valid_pack(self):
        df = _scenarios_catalog_df()
        if len(df) == 0:
            pytest.skip("no packs available")
        pack_id = df.iloc[0]["ID"]
        md = _scenarios_detail_md(pack_id)
        assert pack_id in md
        assert "WIN WHEN" in md or "Levels" in md
