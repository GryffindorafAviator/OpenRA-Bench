"""Tests for the non-gamer **Playlist** mode plumbing.

The Playlist tab is the cold-start human-baseline UX (`Playlist`
gradio tab in `app.py`); these tests cover the pure-Python plumbing
that backs it — playlist composition, the jargon-substitution
dictionary, the auto-advance state machine, and the progress bar.

Live-engine bits are guarded by the conftest engine-token list (the
playlist runtime opens an `InteractiveSession` exactly like the
24-pack `human_study`, so the same skip rule covers it for free)."""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.playlist import (
    AUTO_ADVANCE_WAIT_SECONDS,
    JARGON_TO_PLAIN,
    NOVICE_PLAYLIST,
    OUTCOME_COLORS,
    PLAYLIST_TOOLS,
    explain_outcome,
    is_valid_player_name,
    needs_build_tool,
    outcome_html,
    playlist_progress,
    playlist_progress_bar,
    playlist_should_advance,
    session_summary_row,
    simplify_objective,
    simplify_text,
)

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


# ── 20-pack curated list ─────────────────────────────────────────────


def test_curated_playlist_is_exactly_20_easy_packs():
    """The non-gamer playlist must be EXACTLY 20 (pack, level) pairs —
    a tester completes it in 60-90 minutes; more is fatigue, less
    leaves the human-baseline data underpowered."""
    assert len(NOVICE_PLAYLIST) == 20
    assert len(set(NOVICE_PLAYLIST)) == 20  # no duplicates
    for pack, level in NOVICE_PLAYLIST:
        # Easy tier only — non-gamer accessibility.
        assert level == "easy", (pack, level)


def test_every_curated_pack_exists_on_disk():
    """A pack id in the playlist must resolve to a real YAML file —
    catches typos at test time, not at gradio-launch time."""
    for pack, _level in NOVICE_PLAYLIST:
        assert (PACKS / f"{pack}.yaml").exists(), pack


def test_curated_playlist_capability_spread():
    """The 20-pack list is stratified — combat / coordination /
    defense / multi-objective. A monoculture is bad UX (non-gamer
    sees the same scenario type 20 times)."""
    prefixes = {pack.split("-", 1)[0] for pack, _ in NOVICE_PLAYLIST}
    # We expect at least 4 distinct family prefixes (combat, coord,
    # def, action / artofwar / proc). The exact composition can shift
    # if a pack is retired; the spread invariant is what we pin.
    assert len(prefixes) >= 4, sorted(prefixes)


# ── Jargon dictionary ────────────────────────────────────────────────


def test_jargon_dict_covers_required_substitutions():
    """The CLAUDE.md / task spec calls out specific jargon every UI
    string must replace. Pin the must-have substitutions here so a
    later refactor can't regress them."""
    required = {
        "mcv": "base builder",
        "harvester": "miner",
        "barracks": "infantry factory",
        "war factory": "tank factory",
        "construction yard": "main base",
    }
    for k, v in required.items():
        assert k in JARGON_TO_PLAIN, k
        assert JARGON_TO_PLAIN[k] == v, (k, JARGON_TO_PLAIN[k], v)


def test_simplify_text_replaces_short_codes_and_phrases():
    """`simplify_text` is the render-time substitution. Both the long
    name ('barracks') and the short code ('barr') resolve, the result
    is lowercased, and case-insensitive matching catches `MCV`."""
    s = "MCV deploys; harvester returns to refinery; build a barracks."
    out = simplify_text(s)
    # MCV → base builder; harvester → miner; refinery → miner depot
    # (via 'ore refinery' is multi-word so partial 'refinery' alone is
    # not in the dict — keep the assertion to the dict's contents).
    assert "MCV" not in out
    assert "base builder" in out
    assert "miner" in out
    assert "infantry factory" in out
    # Short codes match too — pin both axes.
    assert simplify_text("e1") == "rifleman"
    assert simplify_text("2tnk") == "medium tank"
    # Whole-word: 'mcvent' must NOT match `mcv`.
    assert simplify_text("mcvent") == "mcvent"


def test_simplify_text_handles_plural_forms():
    """The per-turn briefing line `Funds: ... | Harvesters: 0` would
    otherwise leak un-substituted `Harvesters` even though `harvester`
    → `miner` is in the dict. Pin the plural-aware substitution so a
    refactor of the regex can't regress the briefing-line UX (the
    smoke-test friction the Playlist pass uncovered)."""
    # Plural single-word keys round-trip with the plural preserved.
    assert simplify_text("harvesters") == "miners"
    assert simplify_text("Harvesters") == "miners"  # case-insensitive
    assert simplify_text("MCVs") == "base builders"
    assert simplify_text("pillboxes") == "guard towers"
    # The actual briefing line shape — the repro the smoke-test caught.
    out = simplify_text(
        "Funds: $0 (cash=$0 + ore=$0) | Power: +0 | Harvesters: 0"
    )
    assert "Harvesters" not in out
    assert "miners: 0" in out
    # Whole-word still applies to the plural — `mcvents` must NOT match
    # `mcv` even though the suffix is `s`.
    assert simplify_text("mcvents") == "mcvents"
    # An identity-passthrough plural key (`engineer` → `engineer`) is
    # still pluralised correctly.
    assert simplify_text("engineers") == "engineers"


def test_simplify_objective_strips_machine_block():
    """The Playlist plain-English objective drops the structured
    `WIN WHEN: …` / `YOU LOSE IF: …` machine block — those go behind
    the 'Details' expand."""
    raw = (
        "Drive your medium tanks east and kill the rocket soldiers.\n"
        "WIN WHEN: units_killed_gte:5 AND within_ticks:2700.\n"
        "YOU LOSE IF: not units_lost_lte:1.\n"
        "You have at most 30 decision turns; acting decisively..."
    )
    out = simplify_objective(raw)
    assert "WIN WHEN" not in out
    assert "YOU LOSE IF" not in out
    assert "decision turns" not in out
    # Plain-English content survives, with substitutions applied.
    assert "tank" in out.lower()
    assert "rocket trooper" in out.lower() or "rocket" in out.lower()


def test_simplify_objective_caps_length():
    """Non-gamer should see ~1-2 sentences. A wall-of-text briefing
    must be truncated."""
    long_raw = "First sentence. " * 200
    out = simplify_objective(long_raw, max_chars=200)
    assert len(out) <= 200


# ── Reduced command palette ─────────────────────────────────────────


def test_playlist_tools_excludes_advanced_verbs():
    """A non-gamer must not see `repair`, `power_down`, `set_primary`,
    `infiltrate`, `fire_superweapon`, `set_stance` in playlist mode."""
    forbidden = {
        "repair", "power_down", "set_primary", "infiltrate",
        "fire_superweapon", "set_stance",
    }
    assert not (set(PLAYLIST_TOOLS) & forbidden)


def test_needs_build_tool_only_for_build_packs():
    """`needs_build_tool` is the gate that hides the build textbox
    unless the pack authored a `build`/`place_building` tool."""
    assert needs_build_tool(["move_units", "attack_unit"]) is False
    assert needs_build_tool(["build", "move_units"]) is True
    assert needs_build_tool(["place_building"]) is True
    assert needs_build_tool(None) is False
    assert needs_build_tool([]) is False


# ── Auto-advance state machine ──────────────────────────────────────


def test_should_advance_only_when_game_done():
    """Mid-game polls never advance — the only escape is a finished
    game whose 5-second wait window has elapsed."""
    # Mid-game: never advances regardless of wall-clock.
    assert playlist_should_advance(False, None, now=0.0) is False
    assert playlist_should_advance(False, 0.0, now=999.0) is False


def test_should_advance_after_wait_window_elapses():
    """After game-over, advance fires only once `wait_seconds` have
    passed since the `done_at` timestamp. Pin both edges of the
    window — just-before stays put, just-after advances."""
    done_at = 100.0
    wait = AUTO_ADVANCE_WAIT_SECONDS
    # Just after game ends — wait not yet elapsed.
    assert playlist_should_advance(True, done_at, now=done_at + 0.1) is False
    # Half-way through the wait — still pending.
    assert (
        playlist_should_advance(True, done_at, now=done_at + wait / 2)
        is False
    )
    # Wait fully elapsed — advance.
    assert (
        playlist_should_advance(True, done_at, now=done_at + wait)
        is True
    )
    # Long after — still True (no upper bound).
    assert (
        playlist_should_advance(True, done_at, now=done_at + 999)
        is True
    )


def test_should_advance_handles_none_done_at():
    """If `done=True` but the timestamp was never recorded
    (defensive — shouldn't happen, but the state machine must not
    crash), we DON'T advance — better to leave the user the manual
    'Skip wait' button than auto-advance from an unknown state."""
    assert playlist_should_advance(True, None, now=0.0) is False


def test_should_advance_custom_wait():
    """Wait-seconds is a parameter, not a constant — a hypothetical
    test mode could set it to 0 to skip the countdown."""
    assert playlist_should_advance(True, 50.0, now=50.0, wait_seconds=0.0) is True


# ── Progress bar ────────────────────────────────────────────────────


def test_progress_fraction_clamped():
    """Out-of-range inputs clamp to [0, 1] rather than crashing."""
    assert playlist_progress(0, 20) == pytest.approx(0.0)
    assert playlist_progress(10, 20) == pytest.approx(0.5)
    assert playlist_progress(20, 20) == pytest.approx(1.0)
    assert playlist_progress(-1, 20) == pytest.approx(0.0)
    assert playlist_progress(99, 20) == pytest.approx(1.0)
    # Zero total — defensive default 0, no division-by-zero.
    assert playlist_progress(5, 0) == pytest.approx(0.0)


def test_progress_bar_text_format():
    """The text bar carries 20 cells + a trailing percent. Empty,
    half, and full are pinned — the cell glyphs are part of the UX
    contract, not arbitrary."""
    assert playlist_progress_bar(0, 20).startswith("░" * 20)
    assert "0%" in playlist_progress_bar(0, 20)
    assert "50%" in playlist_progress_bar(10, 20)
    assert "100%" in playlist_progress_bar(20, 20)
    full = playlist_progress_bar(20, 20)
    assert full.startswith("▮" * 20)


# ── Plain-English outcome explanation (B9 nit #1) ───────────────────


def test_explain_outcome_win_is_objective_reached():
    """`win` always renders as 'You reached the objective.' regardless
    of the optional signals — pinning the constant string here keeps
    the non-gamer wording stable across UI refactors."""
    assert explain_outcome("win") == "You reached the objective."
    # Optional signals don't change the win branch.
    assert (
        explain_outcome("win", turn=10, max_turns=10, own_units=0)
        == "You reached the objective."
    )


def test_explain_outcome_loss_branches_pick_most_specific():
    """Loss explanations are most-specific first: a base destroyed +
    timer expired loss reads as 'last unit destroyed' / 'base
    destroyed' — never the generic timer line."""
    # No own units → unit-destroyed branch wins over timer.
    assert (
        explain_outcome("loss", turn=20, max_turns=20, own_units=0)
        == "Your last unit was destroyed."
    )
    # Has units, no base → base-destroyed branch.
    assert (
        explain_outcome(
            "loss", turn=10, max_turns=20, own_units=2, has_base=False
        )
        == "Your base was destroyed."
    )
    # Has units, has base, deadline expired → timer branch.
    assert (
        explain_outcome(
            "loss", turn=20, max_turns=20, own_units=3, has_base=True
        )
        == "You ran out of time."
    )
    # No optional signals — generic fallback.
    assert (
        explain_outcome("loss")
        == "The objective was not reached in time."
    )


def test_explain_outcome_draw_is_no_winner():
    """Draws are rare under the bench bar but the helper still emits a
    non-gamer-friendly sentence rather than a bare token."""
    assert explain_outcome("draw") == "The game ended without a winner."
    # Empty / None outcome falls through to the draw branch.
    assert explain_outcome("") == "The game ended without a winner."


# ── Coloured outcome chip (B9 nit #2) ────────────────────────────────


def test_outcome_html_wraps_known_outcomes_with_palette_color():
    """`win` / `loss` / `draw` each get a coloured `<span>`; the chip
    text is the upper-cased token so the on-screen reading is the
    same as the legacy `**LOSS**` markdown — only the visual contrast
    changes."""
    for outcome in ("win", "loss", "draw"):
        html = outcome_html(outcome)
        assert outcome.upper() in html
        assert "<span" in html
        assert OUTCOME_COLORS[outcome] in html


def test_outcome_html_unknown_outcome_falls_back_to_plain():
    """An unrecognised outcome string returns the upper-cased token
    without HTML — defensive for a future outcome value (e.g.
    'aborted') we haven't pinned a colour for yet."""
    assert outcome_html("aborted") == "ABORTED"
    # Empty / None defaults to the draw chip (matches the rest of the
    # helper family's None-tolerance).
    assert "DRAW" in outcome_html("")


def test_outcome_colors_palette_uses_distinct_hexes():
    """Pin the palette: red-orange for loss, green for win, gray for
    draw. A future palette nudge (e.g. theming) is fine, but the
    distinctness invariant — three different visual chips — is the
    UX contract."""
    assert len({OUTCOME_COLORS["win"], OUTCOME_COLORS["loss"],
                OUTCOME_COLORS["draw"]}) == 3


# ── Player-name validation (B9 nit #3) ──────────────────────────────


def test_is_valid_player_name_rejects_empty_and_whitespace():
    """The Start button is gated on this validator. Empty input,
    None, and whitespace-only input must all return False so the
    silent "anon" fallback never gets recorded against a tester who
    didn't type their name."""
    assert is_valid_player_name(None) is False
    assert is_valid_player_name("") is False
    assert is_valid_player_name("   ") is False
    assert is_valid_player_name("\t\n") is False


def test_is_valid_player_name_accepts_real_names():
    """Single-character names pass (the spec is ≥1 non-whitespace
    char). Whitespace padding is tolerated (a paste-trim case)."""
    assert is_valid_player_name("A") is True
    assert is_valid_player_name("Alex") is True
    assert is_valid_player_name("  Alex  ") is True
    assert is_valid_player_name("张三") is True  # non-ASCII still valid


def test_session_summary_row_shape():
    """The session-end summary uses the same row schema across the
    playlist UI and the playback manifest — pin the column names so
    a downstream consumer (results.csv export) doesn't break silently.
    """
    row = session_summary_row(
        idx=4, pack="combat-focus-fire-priority", level="easy",
        outcome="win", turns=12, max_turns=30,
    )
    assert row == {
        "Game": 5,
        "Scenario": "combat-focus-fire-priority",
        "Level": "easy",
        "Outcome": "WIN",
        "Turns": 12,
        "Max Turns": 30,
    }
