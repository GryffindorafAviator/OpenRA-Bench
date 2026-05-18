"""Adversarial ladder rating (the 1v1 spotlight metric).

Each `adversarial-*` pack is a 3-rung ladder (easy → medium → hard) of
increasing reactive-opponent strength. A model's **ladder rating** on a
pack is the number of rungs cleared *contiguously from the bottom* — a
monotone difficulty signal that complements the Elo (which ranks models
head-to-head on shared rungs via `pairwise.pairwise_elo`).

Pure + deterministic; the live opponent is the engine's reactive force
today, with a documented swap-in to model-vs-model once the engine
exposes an enemy command channel (see pairwise.py / task #3).
"""

from __future__ import annotations

RUNGS: tuple[str, ...] = ("easy", "medium", "hard")


def ladder_rating(outcomes: dict[str, str]) -> int:
    """Rungs cleared contiguously from easy. A rung is cleared iff its
    outcome == "win". easy lost → 0; easy+medium won, hard lost → 2."""
    n = 0
    for r in RUNGS:
        if outcomes.get(r) == "win":
            n += 1
        else:
            break
    return n


def is_adversarial_episode(ep: dict) -> bool:
    return ep.get("capability") == "adversarial"


def ladder_ratings(stats: dict) -> dict[str, int]:
    """Per adversarial pack → ladder rating, from a run_eval stats dict.

    `cell` is "<pack>:<level>"; only the public split counts (held-out
    seeds are anti-memorization, not ladder progression). When a rung
    ran multiple seeds it is cleared only if it was won on *every*
    seed (no lucky-seed promotion)."""
    rungs: dict[str, dict[str, list[str]]] = {}
    for e in stats.get("episodes", []):
        if not is_adversarial_episode(e) or e.get("split", "public") != "public":
            continue
        pack, _, level = str(e.get("cell", "")).rpartition(":")
        if not pack or level not in RUNGS:
            continue
        rungs.setdefault(pack, {}).setdefault(level, []).append(
            e.get("outcome", "?")
        )
    out: dict[str, int] = {}
    for pack, by_level in rungs.items():
        collapsed = {
            lv: ("win" if outs and all(o == "win" for o in outs) else "loss")
            for lv, outs in by_level.items()
        }
        out[pack] = ladder_rating(collapsed)
    return out


def adversarial_summary(stats: dict) -> dict:
    """Spotlight roll-up: per-pack ratings + the headline mean rating
    (0–3) across adversarial packs played."""
    ratings = ladder_ratings(stats)
    mean = round(sum(ratings.values()) / len(ratings), 4) if ratings else 0.0
    return {
        "ladder_ratings": ratings,
        "mean_ladder_rating": mean,
        "packs": sorted(ratings),
        "max_rung": len(RUNGS),
    }
