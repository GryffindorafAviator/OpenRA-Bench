"""Human-baseline study — the apples-to-apple human reference.

External tournament replays cannot be scored on this benchmark: a real
RTS game is a different engine, continuous-time, a different action
space, and not one of the constructed tasks. The ONLY comparable human
data comes through the Play tab — same engine, scenario, observation,
tool API, win predicate, and `Playback` format the models use.

This module defines a fixed, stratified 24-pack subset (one pack per
scenario family, difficulty spread 8 easy / 8 medium / 8 hard) and the
three conditions a player plays each under:

* `vision-fog`   — the canonical fogged minimap (normal play).
* `vision-clear` — no fog (engine `reveal_map`): the fog-penalty pair.
* `handoff-bad`  — the player inherits a losing position (a `stall`
                   prefix of `HANDOFF_K` turns), then plays on — the
                   recovery / freeze comparison.

A per-player counterbalanced playlist walks all 24 x 3 = 72 cells.
"""

from __future__ import annotations

import random
from typing import Any

# Frozen subset — one representative pack per scenario family, levels
# spread across difficulty. Stratified by family (not by the lopsided
# `meta.capability` tag). Keep this list STABLE — it pins the baseline.
STUDY_SUBSET: list[tuple[str, str]] = [
    ("combat-pincer-coordination", "easy"),
    ("econ-overflow-to-silos", "medium"),
    ("def-pre-position-mobile-reserve", "hard"),
    ("build-rally-point-management", "easy"),
    ("scout-detect-incoming-army", "medium"),
    ("lh-opening-to-defense-to-counter", "hard"),
    ("proc-only-defend-no-attack", "easy"),
    ("mfb-supply-line-link-between-bases", "medium"),
    ("rob-objective-shift-with-or-clause", "hard"),
    ("coord-mutual-support", "easy"),
    ("tp-rush-objective-very-fast", "medium"),
    ("mcv-deploy-relocate-under-pressure", "hard"),
    ("artofwar-lure-the-tiger", "easy"),
    ("economy-harvest-timebox", "medium"),
    ("perception-frontier-reading", "hard"),
    ("strategy-trilemma", "easy"),
    ("tech-production-planning", "medium"),
    ("expansion-balanced-2-base-defended", "hard"),
    ("mid-economy-under-fire", "easy"),
    ("strict-sequence", "medium"),
    ("action-sequenced-execution", "hard"),
    ("adv-rps-counter-pick", "easy"),
    ("coordination-staggered-window", "medium"),
    ("maint-sell-and-recoup-cash", "hard"),
]

STUDY_CONDITIONS: tuple[str, ...] = ("vision-fog", "vision-clear", "handoff-bad")

# Prefix length for the `handoff-bad` condition — the player inherits a
# game already `HANDOFF_K` observe-only (stall) turns deep.
HANDOFF_K = 3

# fog_mode each condition compiles the scenario under.
_CONDITION_FOG = {
    "vision-fog": "vision",
    "vision-clear": "vision-clear",
    "handoff-bad": "vision",
}


def study_playlist(player_seed: int = 0) -> list[tuple[str, str, str]]:
    """The 72-cell playlist — every (pack, level) x condition — in a
    per-player counterbalanced (deterministically shuffled) order, so
    condition ordering is not confounded across players."""
    cells = [
        (pack, level, cond)
        for pack, level in STUDY_SUBSET
        for cond in STUDY_CONDITIONS
    ]
    random.Random(player_seed).shuffle(cells)
    return cells


def open_study_session(
    pack: str,
    level: str,
    condition: str,
    player: str,
    seed: int = 1,
    playback_root: Any = None,
):
    """Open an `InteractiveSession` for one study cell, configured for
    `condition`. For `handoff-bad` the engine is advanced `HANDOFF_K`
    observe-only turns BEFORE the player takes over — so the player
    inherits a real deficit, exactly like the model handoff ablation.

    The run persists to the standard `Playback` format (apples-to-apple
    with model runs); `playback_root` defaults to a per-condition dir
    so the condition is recoverable from the path."""
    from pathlib import Path

    from .human_labeling import InteractiveSession

    if condition not in STUDY_CONDITIONS:
        raise ValueError(f"unknown study condition {condition!r}")
    if playback_root is None:
        playback_root = Path("playback/human_study") / condition

    sess = InteractiveSession.from_pack(
        pack, level, seed,
        record=True, playback_root=playback_root, player=player,
        fog_mode=_CONDITION_FOG[condition],
    )
    if condition == "handoff-bad":
        # Stall prefix — the player inherits the resulting losing board.
        for _ in range(HANDOFF_K):
            if sess.done:
                break
            sess.submit_turn([])
    return sess
