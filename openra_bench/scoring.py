"""Episode scoring + Perception/Reasoning/Action diagnostics.

Two outputs per episode:

1. A **composite scalar** in [0,1] using OpenRA-RL-Training's reward-
   weight *schema* (`DEFAULT_REWARD_WEIGHTS`, overridable per scenario
   via `scenario.reward`). The Rust env hardcodes reward to 0.0, so each
   dimension's *value* is computed here from adapter-derived signals —
   not parsed from completions like Training's `reward_funcs`. Keeping
   the weight keys/semantics identical means scenario authors tune the
   same dials Training uses.

2. **P/R/A diagnostics** — a sub-score for each link of the
   Perception → Reasoning → Action chain so a low composite points at
   the *broken link*, which is the bench's reason to exist:

     perception : did it read the spatial state and steer sensing into
                  the unknown? (exploration efficiency, target sighting)
     reasoning  : given what it saw, did it form a plan that achieved
                  the objective in time? (outcome × deadline efficiency,
                  search non-wandering)
     action     : did it emit valid, non-empty commands the engine
                  accepted? (1 - warn_rate, non-idle rate)

These Phase-0 diagnostics are *behavioural proxies*; the hooks for
ground-truth probes (forced state-readback, optimal-plan distance) are
called out inline and are the deeper extension of task #2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from openra_rl_training.training.reward_funcs import DEFAULT_REWARD_WEIGHTS

from .eval_core import EpisodeResult
from .scenarios.schema import CompiledLevel


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class ScoreCard:
    composite: float            # weighted scalar in [0,1]
    outcome: str                # win | draw | loss
    perception: float           # P/R/A link sub-scores, each in [0,1]
    reasoning: float
    action: float
    dimensions: dict            # per-dimension value (pre-weight) in [0,1]
    weights: dict               # weights actually used (scenario or default)
    weakest_link: str           # "perception" | "reasoning" | "action"
    notes: list

    def to_dict(self) -> dict:
        return asdict(self)


def _dimension_values(compiled: CompiledLevel, res: EpisodeResult) -> dict:
    """Map adapter signals -> Training reward dimensions, each in [0,1].

    Only dimensions the Rust env can ground are populated; economy /
    production / disruption stay 0 until Phase 3 (and carry 0 weight in
    DEFAULT_REWARD_WEIGHTS anyway, so they don't distort the composite).
    """
    s = res.signals
    max_ticks = max(1, compiled.scenario.termination.max_ticks)
    tick_frac = _clamp(s.game_tick / max_ticks)

    outcome = {"win": 1.0, "draw": 0.5, "loss": 0.0}[res.outcome]
    # exploration: % of map revealed (0-100 -> 0-1).
    exploration = _clamp(s.explored_percent / 100.0)
    # discovery: enemy units + buildings sighted, soft-capped at 5 contacts.
    discovery = _clamp((len(s.enemies_seen_ids) + len(s.enemy_buildings_seen_ids)) / 5.0)
    # combat: kills net of losses, soft-capped at 5.
    combat = _clamp((s.units_killed - s.units_lost) / 5.0)
    # survival: fewer losses is better (soft-capped at 5).
    survival = _clamp(1.0 - s.units_lost / 5.0)
    # tempo: acted with intent rather than burning the clock idle.
    tempo = _clamp(1.0 - tick_frac) if res.outcome == "win" else _clamp(0.5 - 0.5 * tick_frac)
    # format/action validity: engine-accepted command fraction.
    acceptance = (
        1.0
        if res.actions_issued == 0
        else _clamp(1.0 - res.actions_warned / res.actions_issued)
    )
    return {
        "outcome": outcome,
        "exploration": exploration,
        "discovery": discovery,
        "combat": combat,
        "survival": survival,
        "tempo": tempo,
        "format": acceptance,
        # not grounded by Rust yet:
        "economy": 0.0,
        "density": 0.0,
        "disruption": 0.0,
    }


def _weights(compiled: CompiledLevel) -> dict:
    """Scenario reward overrides on top of the Training default schema."""
    w = dict(DEFAULT_REWARD_WEIGHTS)
    w.update(compiled.scenario.reward or {})
    return w


def _composite(values: dict, weights: dict) -> float:
    total_w = sum(weights.get(k, 0.0) for k in values) or 1.0
    return _clamp(sum(values[k] * weights.get(k, 0.0) for k in values) / total_w)


def _pra_diagnostics(compiled: CompiledLevel, res: EpisodeResult, dims: dict) -> dict:
    """Behavioural proxies for each chain link (Phase 0)."""
    s = res.signals
    max_ticks = max(1, compiled.scenario.termination.max_ticks)
    tick_frac = _clamp(s.game_tick / max_ticks)

    # PERCEPTION — did it sense the unknown? Exploration achieved, plus a
    # bonus for actually sighting contacts (it looked where things were).
    # TODO(task#2 deep): replace with forced state-readback probe vs
    # ground-truth obs (perception independent of acting well).
    perception = _clamp(0.7 * dims["exploration"] + 0.3 * dims["discovery"])

    # REASONING — given perception, did the plan achieve the objective in
    # time, without aimless wandering (discovery per unit explored)?
    # TODO(task#2 deep): distance-to-optimal-plan (path length, target
    # ordering) given the scenario's known solution.
    win = res.outcome == "win"
    efficiency = _clamp(1.0 - tick_frac)
    # focus = did it convert what it saw into contacts (vs wandering)?
    focus = (
        _clamp(dims["discovery"] / dims["exploration"])
        if dims["exploration"] > 0
        else 0.0
    )
    # Floor on the non-win branch so that when *perception* is the
    # bottleneck (near-zero coverage), reasoning is not spuriously the
    # minimum — reasoning can only be judged on what was actually sensed.
    reasoning = _clamp(
        (0.6 + 0.4 * efficiency) if win else (0.2 + 0.5 * focus + 0.1 * dims["exploration"])
    )

    # ACTION — valid, non-empty, accepted commands.
    acceptance = dims["format"]
    non_idle = (
        _clamp(res.actions_issued / max(res.turns, 1) / 1.0) if res.turns else 0.0
    )
    action = _clamp(0.7 * acceptance + 0.3 * min(non_idle, 1.0))
    return {"perception": perception, "reasoning": reasoning, "action": action}


def score_episode(compiled: CompiledLevel, res: EpisodeResult) -> ScoreCard:
    dims = _dimension_values(compiled, res)
    weights = _weights(compiled)
    composite = _composite(dims, weights)
    pra = _pra_diagnostics(compiled, res, dims)

    weakest = min(pra, key=pra.get)
    notes: list[str] = []
    if res.outcome != "win":
        notes.append(f"objective not met ({res.outcome}); weakest link: {weakest}")
    if res.actions_issued and res.actions_warned / res.actions_issued > 0.25:
        notes.append(
            f"high invalid-action rate "
            f"{res.actions_warned}/{res.actions_issued} → action problem"
        )
    if dims["exploration"] < 0.15 and compiled.meta.capability != "action":
        notes.append("very low map coverage → perception/scouting problem")

    return ScoreCard(
        composite=round(composite, 4),
        outcome=res.outcome,
        perception=round(pra["perception"], 4),
        reasoning=round(pra["reasoning"], 4),
        action=round(pra["action"], 4),
        dimensions={k: round(v, 4) for k, v in dims.items()},
        weights=weights,
        weakest_link=weakest,
        notes=notes,
    )
