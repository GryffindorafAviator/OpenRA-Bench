"""Pairwise adversarial evaluation + Elo.

The project goal includes evaluating models "on pairwise adversarial
conditions", not only fixed-scenario absolute scores. True co-located
1v1 (both sides model-driven in one map) is the deeper engine feature
(task #3); this module delivers the tractable, immediately-useful form:
**comparative** pairwise ranking — every model plays the *same*
controlled scenarios, and models are ranked by head-to-head composite
on shared cells via Elo. Pure + deterministic (no engine in
`pairwise_elo`), so it is fully unit-testable; `run_pairwise` wires it
to the live evaluator.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

from .run_eval import AgentFactory, evaluate

# Per model: {cell -> mean composite over that cell's seeds}.
ModelCellScores = dict[str, dict[str, float]]

ELO_K = 32.0
ELO_START = 1000.0
_TIE_EPS = 1e-6


def _cell_means(stats: dict) -> dict[str, float]:
    """Mean composite per cell from a run_eval stats dict (public split)."""
    by: dict[str, list[float]] = {}
    for e in stats.get("episodes", []):
        if e.get("split", "public") != "public":
            continue
        by.setdefault(e["cell"], []).append(e["composite"])
    return {c: sum(v) / len(v) for c, v in by.items() if v}


def pairwise_elo(scores: ModelCellScores) -> dict:
    """Deterministic Elo from comparative per-cell composites.

    For each unordered model pair, the match score = fraction of shared
    cells where A strictly beats B (ties = 0.5). One Elo update per
    pair; pairs processed in sorted order so the result is fully
    deterministic and order-independent given the input.
    """
    models = sorted(scores)
    elo = {m: ELO_START for m in models}
    matrix: dict[str, dict[str, float]] = {m: {} for m in models}

    for a, b in itertools.combinations(models, 2):
        shared = sorted(set(scores[a]) & set(scores[b]))
        if not shared:
            continue
        wins = 0.0
        for c in shared:
            da = scores[a][c] - scores[b][c]
            if da > _TIE_EPS:
                wins += 1.0
            elif abs(da) <= _TIE_EPS:
                wins += 0.5
        sa = wins / len(shared)  # A's match score in [0,1]
        matrix[a][b] = round(sa, 4)
        matrix[b][a] = round(1.0 - sa, 4)
        ea = 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
        elo[a] += ELO_K * (sa - ea)
        elo[b] += ELO_K * ((1.0 - sa) - (1.0 - ea))

    ranked = sorted(models, key=lambda m: (-elo[m], m))
    return {
        "elo": {m: round(elo[m], 1) for m in models},
        "rank": {m: i + 1 for i, m in enumerate(ranked)},
        "matrix": matrix,  # matrix[a][b] = A's match score vs B
        "shared_cells": {
            f"{a}|{b}": len(set(scores[a]) & set(scores[b]))
            for a, b in itertools.combinations(models, 2)
        },
    }


def run_pairwise(
    packs: list[Path],
    levels: list[str],
    seeds: list[int],
    agents: dict[str, AgentFactory],
) -> dict:
    """Run each named model over the same packs/levels/seeds and return
    the pairwise Elo ranking + per-model cell scores."""
    if len(agents) < 2:
        raise ValueError("pairwise eval needs >= 2 models")
    scores: ModelCellScores = {}
    per_model_stats: dict[str, dict] = {}
    for name, factory in agents.items():
        st = evaluate(packs, levels, seeds, agent_factory=factory)
        per_model_stats[name] = st
        scores[name] = _cell_means(st)
    return {
        "pairwise": pairwise_elo(scores),
        "cell_scores": scores,
        "per_model_overall": {
            n: s.get("overall", {}) for n, s in per_model_stats.items()
        },
    }
