"""YAML-referenceable scripted opponents (the enemy "bot generator").

A scenario pack can give the enemy a fixed, deterministic, map-agnostic
behaviour instead of only static actors + a stance:

    enemy: {faction: soviet, bot_type: hunt}   # or bot: hunt

The behaviour runs ENGINE-side (openra-sim `scripted_bot`), because the
Python boundary can neither see fogged enemy ids nor command the enemy
player. This module is the bench-facing surface: the catalogue of
behaviours + compile-time validation so an unknown name fails fast
(mirrors mapgen). `bot_type` is the field that survives the training
schema's serialization; `bot` is an accepted alias.

Behaviours (all derive targets from live world state — no map coords):
  hunt    — each unit attacks the nearest agent unit; pursues.
  rusher  — relentless concentrated charge at the agent's mass; fast.
  patrol  — units oscillate around their own spawn; engage intruders.
  turtle  — hold the spawn position; return fire only.
  guard   — leashed defender: holds its post; lunges at a
            foe only within an aggro radius and snaps back
            past a leash (a decoy can bait it off post).
  raider  — worker-priority harasser: each unit attacks the
            nearest agent harvester; falls back to nearest
            combat actor only when no harvesters are alive
            (the faithful SC2-style worker harass).
"""

from __future__ import annotations

from typing import Any

BEHAVIORS: frozenset[str] = frozenset(
    {"hunt", "rusher", "patrol", "turtle", "guard", "raider"}
)


def read_enemy_bot(enemy: Any) -> str | None:
    """Extract the bot behaviour from an `enemy:` block (dict), honouring
    `bot_type` (survives serialization) or the `bot` alias. Empty/missing
    ⇒ None (stance-only reactive enemy, the default)."""
    if not isinstance(enemy, dict):
        return None
    raw = enemy.get("bot_type") or enemy.get("bot")
    if raw is None:
        return None
    s = str(raw).strip().lower()
    return s or None


def validate_enemy_bot(enemy: Any) -> str | None:
    """Compile-time gate: a declared enemy bot must name a known
    behaviour, else the pack fails at load (not mid-eval)."""
    b = read_enemy_bot(enemy)
    if b is not None and b not in BEHAVIORS:
        raise ValueError(
            f"unknown enemy bot {b!r}; known behaviours: "
            f"{sorted(BEHAVIORS)}"
        )
    return b
