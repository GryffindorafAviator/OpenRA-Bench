"""Leaderboard data layer for run_eval reports.

Pure + file-backed (JSONL append store), so it is fully unit-testable
without launching Gradio. The Gradio app renders `build_table()`.

A *run* = one model evaluated over packs×levels×seeds (the dict from
`run_eval.evaluate`). `ingest_run` appends one immutable record per run;
`build_table` ranks models by overall composite, keeping each model's
*best* run and surfacing a per-capability (perception/reasoning/action)
breakdown + the dominant weakest link — the bench's reason to exist.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

CAPABILITIES = ("perception", "reasoning", "action")
DEFAULT_STORE = Path(__file__).parent.parent / "data" / "leaderboard.jsonl"
# A run must cover at least this many episodes to be rankable (mirrors
# the existing app.py min-games gate; keeps one-off noise off the board).
MIN_EPISODES = 5


def _capability_breakdown(episodes: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cap in CAPABILITIES:
        eps = [e for e in episodes if e.get("capability") == cap]
        if not eps:
            continue
        n = len(eps)
        out[cap] = {
            "n": n,
            "composite": round(sum(e["composite"] for e in eps) / n, 4),
            "win_rate": round(sum(e["outcome"] == "win" for e in eps) / n, 4),
        }
    return out


def ingest_run(
    stats: dict[str, Any],
    model: str,
    store: Path | str = DEFAULT_STORE,
    extra: dict | None = None,
) -> dict:
    """Append one leaderboard record for a `run_eval.evaluate` result.

    Returns the stored record. Idempotent only in the sense that each
    call appends a new immutable run row (history is preserved).
    """
    eps = stats.get("episodes", [])
    overall = stats.get("overall", {})
    record = {
        "model": model,
        "ts": int(time.time()),
        "episodes": overall.get("n", len(eps)),
        "win_rate": overall.get("win_rate", 0.0),
        "composite": overall.get("composite_mean", 0.0),
        "perception": overall.get("perception_mean", 0.0),
        "reasoning": overall.get("reasoning_mean", 0.0),
        "action": overall.get("action_mean", 0.0),
        "weakest_link_hist": overall.get("weakest_link_hist", {}),
        "by_capability": _capability_breakdown(eps),
        "cells": sorted(stats.get("summary", {}).keys()),
    }
    if extra:
        record["meta"] = extra
    store = Path(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    with open(store, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _load(store: Path) -> list[dict]:
    if not Path(store).exists():
        return []
    rows = []
    for line in Path(store).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partial trailing write
    return rows


def build_table(
    store: Path | str = DEFAULT_STORE, min_episodes: int = MIN_EPISODES
) -> list[dict]:
    """Ranked leaderboard: each model's best (highest-composite) run that
    meets the episode threshold, sorted by composite desc. Deterministic;
    ties broken by win_rate then model name."""
    best: dict[str, dict] = {}
    for r in _load(store):
        if r.get("episodes", 0) < min_episodes:
            continue
        m = r["model"]
        cur = best.get(m)
        if cur is None or r["composite"] > cur["composite"]:
            best[m] = r
    rows = sorted(
        best.values(),
        key=lambda r: (-r["composite"], -r["win_rate"], r["model"]),
    )
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        # Dominant failure mode across the run, for at-a-glance triage.
        h = r.get("weakest_link_hist") or {}
        r["weakest_link"] = (
            Counter(h).most_common(1)[0][0] if h else "n/a"
        )
    return rows
