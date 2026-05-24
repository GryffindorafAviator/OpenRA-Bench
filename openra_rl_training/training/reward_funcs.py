"""GRPO reward functions for OpenRA training.

Each function receives completions (list[str]) and extra kwargs from rollout_func.
Returns list[float] rewards, one per completion.

Per-scenario weighting: When ``scenario_weights`` is present in kwargs
(a list of dicts, one per completion), each function multiplies its base
reward by the scenario-specific weight for its signal.  This lets combat-
focused scenarios boost combat reward while economy scenarios boost economy.
"""
from __future__ import annotations

from collections import defaultdict

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "outcome": 0.50,
    "combat": 0.15,
    "economy": 0.10,
    "tempo": 0.10,
    "density": 0.00,
    "format": 0.05,
    "survival": 0.10,
    "discovery": 0.00,
    "disruption": 0.00,
    "exploration": 0.00,
}


def _apply_weights(
    base: list[float], key: str, scenario_weights: list[dict] | None,
) -> list[float]:
    """Multiply base rewards by per-scenario weight for *key*."""
    default_w = DEFAULT_REWARD_WEIGHTS[key]
    if not scenario_weights:
        return [r * default_w for r in base]
    return [
        r * (scenario_weights[i].get(key, default_w) if i < len(scenario_weights) else default_w)
        for i, r in enumerate(base)
    ]


def _normalize_within_group(rewards: list[float], spawn_groups: list[int]) -> list[float]:
    """Center rewards within each spawn group (Fix A).

    Within each group (same map), subtract the group mean so that GRPO
    advantages measure behavioral differences, not spawn luck.
    Groups with only 1 episode are left unchanged.
    """
    if not spawn_groups or len(spawn_groups) != len(rewards):
        return rewards
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, g in enumerate(spawn_groups):
        groups[g].append((i, rewards[i]))
    result = list(rewards)
    for g, entries in groups.items():
        if len(entries) < 2:
            continue
        vals = [v for _, v in entries]
        gmean = sum(vals) / len(vals)
        for idx, _ in entries:
            result[idx] -= gmean
    return result


def _rank_normalize(values: list[float]) -> list[float]:
    """Map values to [-1, +1] via rank normalization with tie handling.

    Robust to outliers — one amazing episode doesn't compress the rest.
    Guarantees equal spacing: best episode ALWAYS gets +1.0, worst -1.0.
    """
    n = len(values)
    if n < 2:
        return [0.0] * n
    sorted_indices = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[sorted_indices[j + 1]] == values[sorted_indices[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        i = j + 1
    return [2.0 * (r - 1.0) / (n - 1.0) - 1.0 for r in ranks]


def _zscore_batch(values: list[float]) -> list[float]:
    """Rank-normalize within batch.

    Replaces z-score: rank normalization is robust to outliers and
    guarantees even advantage spacing regardless of score distribution.
    """
    return _rank_normalize(values)


def _zscore_per_group(values: list[float], spawn_groups: list[int] | None) -> list[float]:
    """Rank-normalize within each spawn group.

    Each spawn group = different map layout = different "prompt".
    Rank-normalizing per group ensures advantages reflect behavioral
    differences within the SAME conditions, not map difficulty.

    Falls back to global rank normalization if no spawn groups provided.
    """
    if not values or len(values) < 2:
        return values

    if not spawn_groups:
        return _rank_normalize(values)

    groups: dict[int, list[int]] = {}
    for i, g in enumerate(spawn_groups):
        groups.setdefault(g, []).append(i)

    result = list(values)
    for indices in groups.values():
        if len(indices) < 2:
            result[indices[0]] = 0.0
            continue
        group_vals = [values[i] for i in indices]
        ranked = _rank_normalize(group_vals)
        for idx, rank_val in zip(indices, ranked):
            result[idx] = rank_val
    return result


def _neutralize_infra(rewards: list[float], kwargs: dict) -> list[float]:
    """Replace infra-failure and tool-call-failure episode rewards with valid-episode mean.

    DAPO-style dynamic sampling (arXiv:2503.14476 Section 3.1): episodes
    that failed due to infrastructure issues (game server crash, vLLM 500
    errors) or tool call degeneration (model produced gibberish instead of
    tool calls) get their reward set to the batch mean of valid episodes.
    After GRPO normalization: advantage = (mean - mean) / std = 0,
    so these episodes contribute zero gradient.
    """
    infra = kwargs.get("infra_failure", [])
    tool_fail = kwargs.get("tool_call_failure", [])
    n = len(rewards)
    # Build combined failure mask
    failed = [False] * n
    for i in range(n):
        if (i < len(infra) and infra[i]) or (i < len(tool_fail) and tool_fail[i]):
            failed[i] = True
    if not any(failed):
        return rewards
    valid = [r for r, f in zip(rewards, failed) if not f]
    if not valid:
        return rewards  # all failed — nothing to anchor on
    vmean = sum(valid) / len(valid)
    return [vmean if failed[i] else r for i, r in enumerate(rewards)]


def reward_outcome(completions: list[str], **kwargs) -> list[float]:
    """Terminal game outcome: +1.0 win, -1.0 lose, 0.0 draw/incomplete."""
    outcomes = kwargs.get("outcome", [])
    if not outcomes:
        base = [0.0] * len(completions)
    else:
        mapping = {"win": 1.0, "lose": -1.0, "draw": 0.0}
        base = [mapping.get(o, 0.0) for o in outcomes]
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "outcome", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_combat(completions: list[str], **kwargs) -> list[float]:
    """Combat efficiency from the 8-dim reward vector."""
    scores = kwargs.get("combat_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "combat", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_economy(completions: list[str], **kwargs) -> list[float]:
    """Economic performance from the 8-dim reward vector."""
    scores = kwargs.get("economy_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "economy", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_tempo(completions: list[str], **kwargs) -> list[float]:
    """Action efficiency — fewer redundant actions = higher reward.

    Tempo IS spawn-correlated (r=0.74 with discovery in Sprint scenario):
    closer spawns → less travel time → better tempo. Apply spawn-group
    normalization to isolate the behavioral component.
    """
    scores = kwargs.get("tempo_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "tempo", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_density(completions: list[str], **kwargs) -> list[float]:
    """Action density — parallel utilization of controllable resources.

    Measures how many distinct objectives are pursued per turn relative
    to available units. Independent of tempo (which measures activity/idle).
    3 units with 3 separate commands to 3 places → high density.
    3 units with 1 blob command → low density.
    """
    scores = kwargs.get("density_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "density", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """Format compliance — fraction of turns with valid structured action syntax."""
    scores = kwargs.get("format_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "format", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_survival(completions: list[str], **kwargs) -> list[float]:
    """Unit HP preservation — discourages suicide attacks."""
    scores = kwargs.get("survival_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "survival", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_discovery(completions: list[str], **kwargs) -> list[float]:
    """Discovery reward — accumulated intelligence score from scouting.

    The game engine awards 0.05 per new enemy unit sighting + bonuses for
    buildings (0.2 production, 0.5 base).  Values are accumulated across all
    ticks and clamped to [0, 1].
    """
    scores = kwargs.get("discovery_score", [])
    if not scores:
        base = [0.0] * len(completions)
    else:
        base = [min(max(float(s), 0.0), 1.0) for s in scores]
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "discovery", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_disruption(completions: list[str], **kwargs) -> list[float]:
    """Strategic sabotage — destroying enemy power, production, tech."""
    scores = kwargs.get("disruption_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "disruption", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)


def reward_exploration(completions: list[str], **kwargs) -> list[float]:
    """Map exploration percentage — rewards fog-of-war clearing."""
    scores = kwargs.get("exploration_score", [])
    base = [float(s) for s in scores] if scores else [0.0] * len(completions)
    normalized = _zscore_per_group(base, kwargs.get("spawn_group"))
    weighted = _apply_weights(normalized, "exploration", kwargs.get("scenario_weights"))
    return _neutralize_infra(weighted, kwargs)
