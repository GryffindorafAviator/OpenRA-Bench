"""The human-baseline study harness — the apples-to-apple human
reference: a fixed 24-pack subset, three conditions, a per-player
counterbalanced playlist, run through the same Play-tab `Playback`
pipeline the models use."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.human_study import (HANDOFF_K, STUDY_CONDITIONS,
                                      STUDY_SUBSET, open_study_session,
                                      study_playlist)

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def test_subset_is_24_packs_that_all_exist():
    assert len(STUDY_SUBSET) == 24
    for pack, level in STUDY_SUBSET:
        assert (PACKS / f"{pack}.yaml").exists(), pack
        assert level in ("easy", "medium", "hard"), (pack, level)
    # difficulty is stratified, not all one level
    levels = {lvl for _, lvl in STUDY_SUBSET}
    assert levels == {"easy", "medium", "hard"}


def test_playlist_is_full_grid_and_counterbalanced():
    pl0 = study_playlist(player_seed=0)
    assert len(pl0) == len(STUDY_SUBSET) * len(STUDY_CONDITIONS) == 72
    # every (pack, level, condition) cell appears exactly once
    assert len(set(pl0)) == 72
    for pack, level, cond in pl0:
        assert (pack, level) in STUDY_SUBSET
        assert cond in STUDY_CONDITIONS
    # different players get a different order (counterbalancing)
    assert study_playlist(1) != pl0


@pytest.mark.parametrize("cond", STUDY_CONDITIONS)
def test_open_study_session_each_condition(cond, tmp_path):
    sess = open_study_session(
        "perception-frontier-reading", "hard", cond,
        player="tester", seed=1, playback_root=tmp_path,
    )
    try:
        rs = sess.render_state()
        if cond == "vision-clear":
            # no fog — engine reveal_map: whole map explored
            assert rs.get("explored_percent", 0) > 99.0
            assert sess.compiled.fog_mode == "vision-clear"
        elif cond == "handoff-bad":
            # the player inherits a HANDOFF_K-turn-deep deficit
            assert sess.turn == HANDOFF_K
            assert sess.compiled.fog_mode == "vision"
        else:  # vision-fog — normal fogged start
            assert sess.turn == 0
            assert rs.get("explored_percent", 100) < 50.0
    finally:
        sess.close()


def test_study_session_persists_playback(tmp_path):
    """A study game must save in the standard Playback format so it is
    apples-to-apple with model runs."""
    sess = open_study_session(
        "perception-frontier-reading", "easy", "vision-fog",
        player="tester", seed=1, playback_root=tmp_path,
    )
    try:
        sess.submit_turn([])  # one observe turn
        assert sess._playback is not None
        assert sess._playback.dir.exists()
    finally:
        sess.close()
