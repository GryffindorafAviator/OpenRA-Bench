"""Non-gamer playlist — the 20-pack curated baseline.

The 24-pack `human_study` (`openra_bench/human_study.py`) is built for
recruited testers who can be expected to read jargon and grind through
72 games. A non-gamer cold-clicking through the Play tab cannot do that
— they need a curated, accessible 20-pack run that finishes in
60-90 minutes WITHOUT requiring RTS knowledge.

This module is the support data for the **Playlist** tab in `app.py`:

* `NOVICE_PLAYLIST` — 20 hand-picked (pack, level) pairs whose
  objective is visually obvious on the minimap and whose tool surface
  fits the reduced palette (move / attack / build / end-turn). Each
  pack chosen because:

  - the objective is "drive your green units to the yellow ring" or
    "kill the visible red enemies" (no jargon decoding required);
  - the tool surface is `move_units` + `attack_unit` (and at most
    `build` for one or two build-tier picks); no `repair`,
    `power_down`, `set_primary`, `infiltrate`, `fire_superweapon`,
    `set_stance`;
  - the timer is generous (≥30 decision turns at easy/medium);
  - the easy-level WIN bar has been verified by an intended scripted
    policy in the existing test suite.

* `JARGON_TO_PLAIN` — the render-time substitution dict that converts
  RTS shorthand ("MCV", "harvester", "barracks") into plain English
  ("base builder", "miner", "infantry factory") for the simplified
  briefing. Applied at the UI layer ONLY — the underlying YAML, the
  engine, the tool API, the playback transcript still use the canonical
  short codes, so a Playlist-mode run is still apples-to-apple with a
  model run on the same pack.

* `simplify_briefing` / `simplify_objective` — apply
  `JARGON_TO_PLAIN` to a string. Used by the Playlist tab to produce
  the 1-2 sentence plain-English objective shown above the minimap.

* `playlist_progress(idx, total)` — turn (i, n) into a 0..1 fraction
  for the progress bar.

Auto-advance state-machine:
* `playlist_should_advance(status, last_done_at, now, wait_seconds)`
  — True if a game is over AND the wait countdown has elapsed.

Everything here is pure-Python, no engine handles required, so the
playlist plumbing tests run even when the Rust wheel is absent.
"""

from __future__ import annotations

import re
from typing import Iterable

# ── The curated 20-pack list ────────────────────────────────────────
# Hand-picked from the 210 active packs for accessibility:
# - visually obvious objective (yellow ring or visible red enemies)
# - reduced tool surface (move + attack only, in most cases)
# - generous timer (≥30 decision turns)
# - easy tier — never medium/hard for a non-gamer
#
# Stratified across 5 capability buckets so the player sees variety:
#   * 8 combat (move tanks, kill visible enemies)
#   * 4 coordination (move multiple groups)
#   * 4 defense (stay near base, kill incoming)
#   * 2 reach-region (drive to yellow ring)
#   * 2 multi-objective (split groups between rings)
#
# Why each pack made the cut (one-liner per pack — also doc'd in the
# tab's preset description):
NOVICE_PLAYLIST: list[tuple[str, str]] = [
    # ── Combat — kill the visible red enemies ────────────────────
    # 4 tanks vs visible static squad; just drive east and shoot.
    ("combat-focus-fire-priority", "easy"),
    # Two groups converge, fight at one point — visible attack target.
    ("combat-pincer-coordination", "easy"),
    # 4 tanks, single visible enemy line — straight-line attack.
    ("combat-flanking-attack", "easy"),
    # 5 tanks east through fire corridor — push through.
    ("combat-formation-tank-wedge", "easy"),
    # 4 tanks vs 2 visible clusters — pick one, then the other.
    ("combat-divide-and-conquer", "easy"),
    # Hold a chokepoint vs incoming — defensive but visible.
    ("combat-hold-chokepoint", "easy"),
    # Skirmish + back off — one engagement, visible enemy.
    ("combat-skirmish-then-disengage", "easy"),
    # 4 tanks retreat and re-engage — visible enemy, simple.
    ("combat-retreat-after-engagement", "easy"),

    # ── Coordination — move multiple groups ──────────────────────
    # 6 tanks west edge → punch east, converge on objective.
    ("coord-mutual-support", "easy"),
    # 3 columns from 3 bearings — pick one ring at a time.
    ("coord-converge-on-target", "easy"),
    # 2 squads cross fire zone — staggered move.
    ("coord-cover-and-move", "easy"),
    # Multi-region: move 2 units into each of 2 yellow rings.
    ("action-multiunit-coordination", "easy"),

    # ── Defense — stay near base, kill incoming ──────────────────
    # Defenders hold base vs probing raiders — wait + shoot.
    ("proc-only-defend-no-attack", "easy"),
    # Heavy assault on base — fall back to evac point.
    ("def-evacuation", "easy"),
    # Waves from multiple directions — distribute defenders.
    ("def-multi-direction", "easy"),
    # Mobile reserve plugs the breach — react to single lane.
    ("def-reinforce-the-breach", "easy"),

    # ── Reach-region — drive to the yellow ring ──────────────────
    # Bait-counter — guard line; pull tigers off the mountain.
    ("artofwar-lure-the-tiger", "easy"),
    # 4 tanks face cluster behind fog — drive east, kill.
    ("combat-attack-from-behind-fog", "easy"),

    # ── Multi-objective easier ───────────────────────────────────
    # Visible counter-attack target — bait then kill.
    ("combat-bait-counter-attack", "easy"),
    # 3 fast jeeps kite a heavy tank — visible single target.
    ("combat-kite-jeep-vs-tank", "easy"),
]


# ── Render-time jargon substitutions ─────────────────────────────────
# RTS shorthand → plain English. Applied at the UI rendering layer
# ONLY (briefings, unit-table type column). The engine, tool API, YAML
# pack files, playback transcripts, and command translator continue to
# use the canonical short codes — so the underlying run is still
# apples-to-apple with a model run on the same scenario.
#
# Keys are matched as whole words/phrases (case-insensitive). Longer
# multi-word keys are applied first so "construction yard" wins over
# "yard". Short-code keys (mcv, harv) are matched as whole words so
# "harvester" doesn't double-substitute.
JARGON_TO_PLAIN: dict[str, str] = {
    # Buildings — full names FIRST so "construction yard" beats "yard".
    "construction yard": "main base",
    "ore refinery": "miner depot",
    "war factory": "tank factory",
    "tank factory": "tank factory",  # passthrough; keeps display stable
    "service depot": "repair shop",
    "advanced power plant": "big power plant",
    "barracks": "infantry factory",
    "soviet barracks": "infantry factory",
    "allied barracks": "infantry factory",
    "ore silo": "ore storage",
    "radar dome": "radar",
    # Units — full names.
    "harvester": "miner",
    "rocket soldier": "rocket trooper",
    "rifle infantry": "rifleman",
    "rifle soldier": "rifleman",
    "grenadier": "grenade trooper",
    "engineer": "engineer",  # passthrough — already plain
    "medium tank": "medium tank",  # passthrough
    "heavy tank": "heavy tank",
    "mammoth tank": "mammoth tank",
    "light tank": "light tank",
    "anti-tank": "anti-tank",
    # Defenses.
    "pillbox": "guard tower",
    "tesla coil": "lightning tower",
    "anti-aircraft": "anti-air",
    "turret": "guard turret",
    # Short codes — whole-word match keeps these from clobbering
    # multi-letter words.
    "mcv": "base builder",
    "harv": "miner",
    "fact": "main base",
    "proc": "miner depot",
    "powr": "power plant",
    "apwr": "big power plant",
    "barr": "infantry factory",
    "tent": "infantry factory",
    "weap": "tank factory",
    "hpad": "helipad",
    "afld": "airfield",
    "fix": "repair shop",
    "dome": "radar",
    "silo": "ore storage",
    "pbox": "guard tower",
    "hbox": "guard tower",
    "tsla": "lightning tower",
    "gun": "guard turret",
    "sam": "anti-air",
    # Unit short codes.
    "1tnk": "light tank",
    "2tnk": "medium tank",
    "3tnk": "heavy tank",
    "4tnk": "mammoth tank",
    "e1": "rifleman",
    "e2": "grenade trooper",
    "e3": "rocket trooper",
    "e6": "engineer",
    "arty": "artillery",
    "apc": "transport",
    "jeep": "jeep",
    "lst": "landing craft",
    "dog": "attack dog",
}


# Pre-compile a single regex: longest keys first, escaped, word-bounded.
# A trailing `s` / `es` is captured as an optional plural suffix so the
# briefing line `Harvesters: 0` simplifies to `miners: 0` (not the
# half-converted `Harvesters: 0` that leaks the unsubstituted jargon
# plural). The plural is reattached to the substitution. Multi-word
# keys (`ore refinery`) don't get a plural suffix — pluralising
# "ore refineries" is not a token boundary the substitution dict targets.
def _compile_jargon_pattern() -> "re.Pattern[str]":
    keys = sorted(JARGON_TO_PLAIN.keys(), key=len, reverse=True)
    # Use lookarounds so short codes (e1, mcv) only match as whole
    # tokens — but multi-word keys like "ore refinery" still match the
    # full phrase even with internal spaces. Capture an optional
    # trailing `s` / `es` for plural forms of single-word keys.
    parts = []
    for k in keys:
        esc = re.escape(k)
        if " " in k:
            parts.append(rf"(?<![A-Za-z0-9_-]){esc}(?![A-Za-z0-9_-])")
        else:
            parts.append(rf"(?<![A-Za-z0-9_-]){esc}(?:es|s)?(?![A-Za-z0-9_-])")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


_JARGON_RE = _compile_jargon_pattern()


def _pluralise(word: str) -> str:
    """Naive English plural for the substituted plain-text replacement.
    `miner` → `miners`; `infantry factory` → `infantry factories`;
    `guard tower` → `guard towers`; `box` → `boxes`. Used only when the
    matched jargon was itself plural — keeps `harvesters` → `miners`."""
    if not word:
        return word
    last = word[-1].lower()
    if word.lower().endswith("y") and len(word) >= 2 and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies"
    if last in ("s", "x", "z") or word.lower().endswith(("sh", "ch")):
        return word + "es"
    return word + "s"


def simplify_text(text: str) -> str:
    """Apply `JARGON_TO_PLAIN` substitutions to a single string. The
    match is case-insensitive, but each replacement is lowercased so
    the briefing reads as plain prose — `MCV` and `mcv` both render as
    `base builder`. A trailing plural `s`/`es` on a single-word key is
    preserved on the substitution (`harvesters` → `miners`) — without
    this, a non-gamer reading the per-turn briefing line `Harvesters: 0`
    would see un-substituted jargon despite the dict promising the
    `harvester` → `miner` translation."""
    if not text:
        return text

    def _sub(m: "re.Match[str]") -> str:
        matched = m.group(0)
        lower = matched.lower()
        if lower in JARGON_TO_PLAIN:
            return JARGON_TO_PLAIN[lower]
        # Plural suffix: peel `es` first, then `s`; whichever leaves a
        # known key is the right split.
        for suffix in ("es", "s"):
            if lower.endswith(suffix):
                stem = lower[: -len(suffix)]
                if stem in JARGON_TO_PLAIN:
                    return _pluralise(JARGON_TO_PLAIN[stem])
        return matched

    return _JARGON_RE.sub(_sub, text)


def simplify_objective(text: str, max_chars: int = 320) -> str:
    """One-or-two-sentence plain-English objective for the Playlist
    panel. Strips the structured `WIN WHEN: …` / `YOU LOSE IF: …`
    suffix `objective_brief` appends — those are jargon-heavy and
    we surface them behind the **Details** expand instead."""
    if not text:
        return text
    plain = simplify_text(text)
    # Drop everything from the structured WIN/LOSE machine block onward.
    for marker in ("WIN WHEN:", "YOU LOSE IF:", "You have at most"):
        idx = plain.find(marker)
        if idx >= 0:
            plain = plain[:idx]
    plain = plain.strip()
    # Keep the first 1-2 sentences; cap at max_chars.
    sentences = re.split(r"(?<=[.!?])\s+", plain)
    out = " ".join(sentences[:2]).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


# ── Reduced command palette — what the Playlist tab exposes ────────
# A non-gamer never needs `repair`, `power_down`, `set_primary`,
# `infiltrate`, `fire_superweapon`, `set_stance`. The Playlist tab
# only exposes:
#   * `move`         — click-and-drag on minimap (already wired)
#   * `attack`       — click an enemy with units selected
#   * `build`        — the textbox already there (only when needed)
#   * `end turn`     — the button at the bottom
PLAYLIST_TOOLS: tuple[str, ...] = (
    "move_units", "attack_unit", "attack_move", "stop", "build",
)


def needs_build_tool(pack_tools: Iterable[str] | None) -> bool:
    """True if the pack's authored `tools:` list contains `build`
    (or a build-tier verb like `place_building`). For Playlist mode
    the Build textbox is hidden unless the pack actually requires it
    — keeps the UI sparse for the (majority) move-and-shoot scenarios.
    """
    if not pack_tools:
        return False
    s = {str(t).lower() for t in pack_tools}
    return bool(s & {"build", "place_building", "cancel_production"})


# ── Auto-advance state machine ───────────────────────────────────────
# A 5-second countdown shown after game-over, then auto-load next pack.

AUTO_ADVANCE_WAIT_SECONDS = 5.0


def playlist_should_advance(
    done: bool,
    done_at: float | None,
    now: float,
    wait_seconds: float = AUTO_ADVANCE_WAIT_SECONDS,
) -> bool:
    """True if the current scenario is over AND the post-game wait has
    elapsed — the auto-advance timer should fire NEXT.

    `done_at` is the wall-clock timestamp at which the session first
    transitioned to `done`. The first poll after game-over passes
    `done_at = now`; subsequent polls compare the elapsed delta. A
    `done=False` always returns False (we never advance mid-game)."""
    if not done or done_at is None:
        return False
    return (now - done_at) >= max(0.0, wait_seconds)


def playlist_progress(idx: int, total: int) -> float:
    """0..1 fraction for the progress bar — `idx` games completed out
    of `total`. Clamped — over-/under-flow is a UI input bug, not a
    crash."""
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, float(idx) / float(total)))


def playlist_progress_bar(idx: int, total: int, width: int = 20) -> str:
    """Plain-text progress bar — '▮▮▮▮▮▮▮▮░░░░░░░░░░░░░  35%'. Used
    when a Markdown pre-formatted block is preferable to a real
    `gr.Progress` (e.g. inside a Markdown block on tab-load before
    any state has been set)."""
    frac = playlist_progress(idx, total)
    filled = int(round(frac * width))
    bar = "▮" * filled + "░" * (width - filled)
    return f"{bar}  {int(round(frac * 100))}%"


def session_summary_row(
    idx: int,
    pack: str,
    level: str,
    outcome: str,
    turns: int,
    max_turns: int,
) -> dict:
    """One row of the session-end summary table — same shape across
    Playlist tab and the underlying playback manifest, so the summary
    is sourced from the same place every time."""
    return {
        "Game": idx + 1,
        "Scenario": pack,
        "Level": level,
        "Outcome": (outcome or "draw").upper(),
        "Turns": int(turns or 0),
        "Max Turns": int(max_turns or 0),
    }
