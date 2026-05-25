"""`python -m openra_bench.run_eval` — run a model over scenario packs.

Runs each (pack, level, seed), scores with `scoring.score_episode`, and
writes an aggregate report (win-rate, mean composite, mean P/R/A, and a
weakest-link histogram per pack/level + overall). The legacy
`evaluate.py` is left untouched (its own tests depend on it); this is
the Rust-stack entrypoint.

Programmatic API (used by tests with an injected agent factory):

    stats = evaluate(packs=[...], levels=["easy"], seeds=[1,2],
                     agent_factory=lambda compiled: my_agent_fn)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .eval_core import run_level, scripted_explore_agent
from .scenarios import load_pack
from .scenarios.loader import PACKS_DIR, compile_level
from .scenarios.schema import CompiledLevel
from .scoring import score_episode

# A bare scripted "stall" — observe only. Useful as a 1v1 opponent
# baseline (the canonical-baseline-side of the LLM-vs-LLM cell) and as
# a 1v1 escape hatch for the `--provider scripted:stall` /
# `--opponent scripted:stall` CLI affordance.
def _stall_agent_fn(_rs, Command):
    return [Command.observe()]


# A bare scripted "rusher" — march every own COMBAT unit toward the
# opposite corner of the map (agent NW → SE; enemy SE → NW). Tuned
# for the canonical adversarial-1v1-macro layout but degenerates
# gracefully on any map (units just charge in the chosen compass
# direction). Coarse but real: it advances ticks of agency on the
# engine. The fairness test uses it as the asymmetry probe — a
# rusher beats stall regardless of which slot it occupies.
_NON_COMBAT_TYPES = frozenset({"harv", "fact", "proc", "powr",
                               "tent", "weap", "syrd", "mcv"})


def _rusher_agent_fn(side: str):
    # Each side rushes a generic FAR-CORNER on the opposite half of
    # the map. Clamped well inside the playable bounds of all three
    # adversarial-1v1-macro rungs (60x60 / 80x80 / 96x72).
    target = (55, 55) if side == "agent" else (10, 10)

    def _fn(render_state, Command):
        cmds = []
        # Prefer attack-on-sight: if any opposing unit/building is
        # already in fog-of-war view, slam every combat unit at the
        # nearest one. Otherwise, keep marching toward the opposite
        # corner — attack-move so units engage en route.
        enemy_positions = render_state.get("enemy_positions") or []
        enemy_buildings = render_state.get("enemy_buildings_seen") or []
        tx, ty = target
        for u in render_state.get("units_summary", []) or []:
            uid = u.get("id")
            if uid is None:
                continue
            if str(u.get("type", "")).lower() in _NON_COMBAT_TYPES:
                continue
            # Target the nearest visible enemy actor/building if any.
            best = None
            ux, uy = u.get("cell_x", 0), u.get("cell_y", 0)
            for e in enemy_positions + enemy_buildings:
                ex, ey = e.get("cell_x", -999), e.get("cell_y", -999)
                if ex < 0:
                    continue
                d2 = (ex - ux) ** 2 + (ey - uy) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, ex, ey, e.get("id"))
            if best is not None and best[3] is not None:
                cmds.append(Command.attack_unit(str(uid), str(best[3])))
            else:
                cmds.append(Command.attack_move(
                    [str(uid)], target_x=tx, target_y=ty,
                ))
        return cmds or [Command.observe()]
    return _fn


# Map a `scripted:<kind>` opponent / provider spec to a side-aware
# controller factory. Used by `--mode 1v1` to wire stall/rusher
# without requiring an LLM provider config. Extend by adding entries
# to this dict; the CLI parses `--provider scripted:stall` /
# `--opponent scripted:rusher` and looks the kind up here.
_SCRIPTED_1V1: dict = {
    "stall": lambda _side: _stall_agent_fn,
    "rusher": lambda side: _rusher_agent_fn(side),
}


def _is_scripted_spec(spec: str | None) -> bool:
    return isinstance(spec, str) and spec.startswith("scripted:")


def _scripted_factory_for_1v1(spec: str, side: str):
    """Resolve `scripted:<kind>` (kind in _SCRIPTED_1V1) to a bare
    agent_fn appropriate for the given side. Raises a clear ValueError
    for unknown kinds so a typo fails fast at CLI parse time."""
    kind = spec.split(":", 1)[1].strip().lower()
    if kind not in _SCRIPTED_1V1:
        raise ValueError(
            f"unknown scripted controller {kind!r}; known kinds: "
            f"{sorted(_SCRIPTED_1V1)}"
        )
    return _SCRIPTED_1V1[kind](side)

# agent_factory: (CompiledLevel) -> agent_fn(render_state, Command)->[Command]
AgentFactory = Callable[[CompiledLevel], Callable]


def _default_agent_factory(provider_cfg) -> AgentFactory:
    if provider_cfg is None:
        return lambda _c: scripted_explore_agent
    from .agent import ModelAgent

    from .game_knowledge import (objective_brief, scenario_primer)
    # The agent's system prompt now defaults to the FULL RA codex +
    # tech tree (every model sees the same reference). The legacy
    # scenario-scoped `unit_codex(codes)` filter is no longer wired
    # in — kept callable for explicit override only.

    def factory(compiled: CompiledLevel):
        agent = ModelAgent(
            provider_cfg,
            allowed_tools=compiled.scenario.tools,
            objective=objective_brief(
                compiled.scenario.description,
                compiled.win_condition,
                compiled.fail_condition,
                compiled.max_turns,
                getattr(compiled, "objective_coords", "exact"),
            ),
            system_extra=scenario_primer(compiled),
            base_map=compiled.scenario.base_map,
            level=compiled.level,
            fog_mode=getattr(compiled, "fog_mode", "vision"),
            agent_faction=getattr(
                getattr(compiled.scenario, "agent", None), "faction", "") or "",
            enemy_faction=getattr(
                getattr(compiled.scenario, "enemy", None), "faction", "") or "",
        )
        return agent.agent_fn

    return factory


def _agg(scores: list) -> dict:
    if not scores:
        return {"n": 0}
    comp = [s.composite for s in scores]
    return {
        "n": len(scores),
        "win_rate": round(sum(s.outcome == "win" for s in scores) / len(scores), 4),
        "composite_mean": round(statistics.fmean(comp), 4),
        "composite_std": round(statistics.pstdev(comp), 4) if len(comp) > 1 else 0.0,
        "perception_mean": round(statistics.fmean(s.perception for s in scores), 4),
        "reasoning_mean": round(statistics.fmean(s.reasoning for s in scores), 4),
        "action_mean": round(statistics.fmean(s.action for s in scores), 4),
        "objective_mean": round(
            statistics.fmean(s.dimensions.get("objective", 0.0) for s in scores), 4
        ),
        # Win-speed: averaged over WINS only (0 when there are none) so
        # it compares how decisively a model wins, not diluted by losses.
        # We exclude wins whose `speed`/`win_turns` were missing in a
        # journal (legacy resumed runs pre-`f9c9c46`): `_shim` fills
        # those with 0.0 as a sentinel for "unknown", and a real win
        # always has speed > 0. Including the sentinel zeros would
        # spuriously inflate the speed gap between fresh and resumed
        # evals (P1.5 / P1.6 in PR #30 review).
        "win_speed_mean": round(
            statistics.fmean(
                [s.speed for s in scores
                 if s.outcome == "win" and s.speed > 0]
            ), 4
        ) if any(s.outcome == "win" and s.speed > 0 for s in scores) else 0.0,
        "win_turns_mean": round(
            statistics.fmean(
                [s.win_turns for s in scores
                 if s.outcome == "win" and s.win_turns > 0]
            ), 2
        ) if any(s.outcome == "win" and s.win_turns > 0 for s in scores) else 0.0,
        "weakest_link_hist": dict(Counter(s.weakest_link for s in scores)),
    }


def _find_win_trajectory(bank: str | Path, cell: str, seed: int) -> str | None:
    """Path to a winning run's messages.json for this cell+seed, scanned
    from a `--handoff-bank` directory of Playback runs — the good-prefix
    source. None when the bank holds no matching win. (Engine actor ids
    are seed-deterministic, so the trajectory must match pack/level/seed
    for a faithful replay.)"""
    base = cell.rsplit(":handoff-", 1)[0]  # "pack:level"
    pack_id, _, level = base.partition(":")
    for mf in sorted(Path(bank).rglob("manifest.json")):
        try:
            m = json.loads(mf.read_text())
        except (ValueError, OSError):
            continue
        if (
            str(m.get("pack_id")) == pack_id
            and str(m.get("level")) == level
            and int(m.get("seed", -1)) == int(seed)
            and str(m.get("outcome")) == "win"
            and (mf.parent / "messages.json").exists()
        ):
            return str(mf.parent / "messages.json")
    return None


def _handoff_wrap(agent, cell: str, seed: int, k: int, bank):
    """Wrap `agent` in a HandoffController for a `:handoff-<kind>` cell.
    Returns (controller, note)."""
    from .handoff import HandoffController, TrajectoryController, stall_policy

    kind = cell.rsplit(":handoff-", 1)[1]
    if kind == "bad":  # losing prefix — the recovery / freeze test
        return HandoffController(stall_policy, agent, k), ""
    if kind == "good":  # winning prefix — capitalize-on-advantage
        traj = _find_win_trajectory(bank, cell, seed) if bank else None
        if traj is None:
            return (
                HandoffController(stall_policy, agent, 0),
                f"no winning trajectory in bank for seed {seed} — ran as base",
            )
        return HandoffController(TrajectoryController(traj), agent, k), ""
    # base — k=0; the model plays the whole episode (baseline passivity).
    return HandoffController(stall_policy, agent, 0), ""


def _git_sha() -> str:
    """Best-effort short git SHA for the journal header `code_version`
    field. Returns '' when git is unavailable or this isn't a checkout
    — the header still serializes, the field is just empty."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _score_path_candidates(playback_root, run_id, safe_model,
                           cell: str, split: str, seed: int) -> list[Path]:
    """Candidate on-disk `score.json` locations for a journaled cell.

    The Playback writer (see `playback.py`) builds dirs under
    `<playback_root>/<run_id>__<safe_model>/<sanitized-cell:split>__seed<N>/`.
    We don't replicate the sanitizer's exact rules here — instead we
    glob for any `score.json` whose parent dir starts with a
    prefix matching the cell+split+seed signature, which is what the
    production sweeps use. Returns ALL matches so the caller can pick
    the first that exists."""
    if not playback_root:
        return []
    root = Path(playback_root) / f"{run_id}__{safe_model}"
    if not root.exists():
        return []
    # Tolerate any sanitizer that swapped ":" / "/" / "|" for "_".
    # Cell+split+seed are deterministic; just glob.
    safe_cell = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{cell}:{split}")
    pattern = f"{safe_cell}*seed{seed}*/score.json"
    return sorted(root.glob(pattern))


def _strict_resume_gate(journal, prior: list[dict],
                        playback_root, run_id: str,
                        safe_model: str, *, progress=None):
    """Verify each journaled cell against its on-disk `score.json`.

    Returns (done_keys, kept_prior, stale_keys). For every prior row:
      * `score.json` missing → DROP (re-run); log a `_journal_stale`
        note via progress if a callback is provided.
      * outcomes disagree → DROP (re-run); log mismatch.
      * both present and agree → KEEP (normal resume).

    `done_keys` is the set of keys we still consider DONE under the
    strict gate; the run loop uses it to filter `tasks`.
    """
    kept: list[dict] = []
    stale: list[str] = []
    done: set[str] = set()
    for r in prior:
        key = r.get("_key")
        if not key:
            continue
        cell = r.get("cell", "")
        split = r.get("split", "public")
        seed = r.get("seed", 0)
        outcome = r.get("outcome")
        # Errors never count as done (mirror the loose-resume path),
        # so the existing done_keys() filter handles them. Strict
        # gate only further checks WIN/LOSS/DRAW cells.
        if outcome == "error":
            continue
        cands = _score_path_candidates(
            playback_root, run_id, safe_model, cell, split, seed,
        )
        sc_path = next((c for c in cands if c.exists()), None)
        if sc_path is None:
            stale.append(f"{key}: missing score.json")
            continue
        try:
            sc = json.loads(sc_path.read_text())
        except Exception:  # noqa: BLE001 — corrupt → re-run
            stale.append(f"{key}: corrupt score.json")
            continue
        if sc.get("outcome") != outcome:
            stale.append(
                f"{key}: journal outcome={outcome!r} disagrees with "
                f"score.json outcome={sc.get('outcome')!r}"
            )
            continue
        kept.append(r)
        done.add(key)
    if stale:
        sys.stderr.write(
            f"[strict-resume] dropping {len(stale)} journaled "
            f"entries (mismatch with on-disk score.json):\n"
        )
        for m in stale[:25]:
            sys.stderr.write(f"  • {m}\n")
        if len(stale) > 25:
            sys.stderr.write(f"  • ... and {len(stale) - 25} more\n")
        sys.stderr.flush()
    return done, kept, stale


def _run_adaptive_pool(tasks: list, run_fn: Callable, record_fn: Callable,
                       initial_concurrency: int) -> None:
    """Threadpool runner with rolling-window error-rate adaptation.

    Policy (per the production-eval spec):
      * Start at `initial_concurrency`.
      * If error rate > 10% over last 20 cells → halve concurrency
        (floor 1). One-shot — won't halve again until the window
        re-fills.
      * If error rate < 2% over 50 cells → restore to the original
        starting concurrency (or step up by 25% if already there).

    Implemented as a refilling worker pool: the pool runs at the
    current `cap`, and a controller thread adjusts cap between
    completions. Concurrency changes are logged to stderr with the
    trigger so a long run's log lets you reconstruct the adaptation
    timeline.
    """
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

    if not tasks:
        return
    n = len(tasks)
    cap = max(1, initial_concurrency)
    original = cap
    window20: deque = deque(maxlen=20)
    window50: deque = deque(maxlen=50)
    pending: list = []
    next_idx = 0
    halved_at_window_fill: int = -1  # last completed-count where we halved

    def _log(msg: str) -> None:
        sys.stderr.write(f"[adaptive-concurrency] {msg}\n")
        sys.stderr.flush()

    with ThreadPoolExecutor(max_workers=max(initial_concurrency, 1)) as ex:
        # Prime the pool up to the current cap.
        while next_idx < n and len(pending) < cap:
            pending.append(ex.submit(run_fn, tasks[next_idx]))
            next_idx += 1

        completed = 0
        while pending:
            done, _not_done = wait(pending, return_when=FIRST_COMPLETED)
            for fu in done:
                pending.remove(fu)
                rec = fu.result()
                completed += 1
                err = 1 if rec.get("outcome") == "error" else 0
                window20.append(err)
                window50.append(err)
                record_fn(rec)

                # Adapt: halve on hot error streak (one-shot per fill).
                if (
                    len(window20) == 20
                    and sum(window20) / 20.0 > 0.10
                    and cap > 1
                    and completed > halved_at_window_fill
                ):
                    new_cap = max(1, cap // 2)
                    _log(
                        f"error rate {sum(window20)}/20 > 10% — "
                        f"halving concurrency {cap} → {new_cap}"
                    )
                    cap = new_cap
                    halved_at_window_fill = completed

                # Adapt: restore on clean 50-cell stretch.
                if (
                    len(window50) == 50
                    and sum(window50) / 50.0 < 0.02
                    and cap < original
                ):
                    new_cap = min(original, max(cap + 1, int(cap * 1.25)))
                    _log(
                        f"error rate {sum(window50)}/50 < 2% — "
                        f"restoring concurrency {cap} → {new_cap}"
                    )
                    cap = new_cap

            # Refill the pool up to the current cap.
            while next_idx < n and len(pending) < cap:
                pending.append(ex.submit(run_fn, tasks[next_idx]))
                next_idx += 1


def evaluate(
    packs: list[Path],
    levels: list[str],
    seeds: list[int],
    provider_cfg=None,
    agent_factory: AgentFactory | None = None,
    held_out_seeds: list[int] | None = None,
    playback_root: str | Path | None = None,
    concurrency: int = 1,
    run_id: str | None = None,
    model: str | None = None,
    journal_path: str | Path | None = None,
    resume: bool = True,
    max_spend_usd: float = 0.0,
    smoke: bool = False,
    dry_run: bool = False,
    report_path: str | Path | None = None,
    progress=None,
    perception_sweep: bool = False,
    handoff_sweep: bool = False,
    handoff_k: int = 3,
    handoff_bank: str | Path | None = None,
    repeats: int = 1,
    full_playback_root: str | Path | None = None,
    strict_resume: bool = False,
    ignore_run_id: bool = False,
    adaptive_concurrency: bool = False,
) -> dict:
    """Run packs×levels×seeds. If `held_out_seeds` is given, those are
    run too and tagged split='held_out'; the report adds
    `overall_held_out` and `generalization_gap` (public composite −
    held-out composite) — the anti-memorization metric the
    generalization literature (Procgen/SMACv2/lmgame-Bench) requires.

    `perception_sweep` expands every pack×level into the 4 perception
    ablation cells (`pack:level:<mode>` for mode in PERCEPTION_MODES —
    vision/structured × fog/no-fog) instead of the raw 3 levels, so one
    run yields the full channel-cost / fog-cost decomposition.

    `handoff_sweep` expands every pack×level into handoff cells
    (`pack:level:handoff-{base,bad,good}`): the model plays the whole
    episode (`base`), or inherits a losing position after a `stall`
    prefix (`bad` — the recovery / freeze-and-panic test), or a winning
    position replayed from a `handoff_bank` trajectory (`good` — the
    capitalize-on-advantage test). `handoff_k` is the prefix length.
    Each record carries a `passivity` stat (observe/stop-only fraction).

    `repeats` runs each (cell, seed) `N` times, varying only model
    nondeterminism (assumes temperature > 0). Records carry a `repeat`
    index 0..N-1, so aggregation can report mean ± CI and `pass^k`
    (all-k wins) alongside `pass@k` — the reliability metric.
    """
    from .resilience import (
        BudgetExceeded,
        CostMeter,
        RateLimiter,
        RunJournal,
        episode_key,
    )

    # One shared cost meter + rate limiter across the whole sweep, so
    # the budget cap and throttle apply globally (not per episode).
    meter = CostMeter(
        getattr(provider_cfg, "price_in_per_m", 0.0),
        getattr(provider_cfg, "price_out_per_m", 0.0),
        max_usd=max_spend_usd,
    )
    limiter = RateLimiter(getattr(provider_cfg, "qps", 0.0) or 0.0)
    if agent_factory is not None:
        factory = agent_factory
    elif provider_cfg is None:
        factory = lambda _c: scripted_explore_agent  # noqa: E731
    else:
        from .agent import ModelAgent
        from .providers import make_provider

        shared = make_provider(
            provider_cfg, rate_limiter=limiter, cost_meter=meter
        )

        from .game_knowledge import (objective_brief, scenario_primer)
        # Full RA codex + tech tree is the default reference (see
        # prompt_v2.system_prompt); no per-scenario filter is needed.

        def factory(compiled: CompiledLevel):
            return ModelAgent(
                provider_cfg,
                allowed_tools=compiled.scenario.tools,
                objective=objective_brief(
                    compiled.scenario.description,
                    compiled.win_condition,
                    compiled.fail_condition,
                    compiled.max_turns,
                    getattr(compiled, "objective_coords", "exact"),
                ),
                provider=shared,
                system_extra=scenario_primer(compiled),
                base_map=compiled.scenario.base_map,
                level=compiled.level,
                fog_mode=getattr(compiled, "fog_mode", "vision"),
                agent_faction=getattr(
                    getattr(compiled.scenario, "agent", None), "faction", "") or "",
                enemy_faction=getattr(
                    getattr(compiled.scenario, "enemy", None), "faction", "") or "",
            ).agent_fn

    # Run/model identity so a single playback root can hold many runs
    # and the viewer can filter run → model → scenario.
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    model = model or getattr(provider_cfg, "model", None) or "agent"
    _safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    skipped: list[str] = []
    held_out_seeds = held_out_seeds or []

    # Build the flat list of independent episodes (each is fully
    # isolated: own RustEnvPool, own agent, own playback dir) so they
    # can run concurrently.
    tasks: list[tuple] = []
    for pack_path in packs:
        pack = load_pack(pack_path)
        # Quarantined packs stay runnable by explicit --packs but never
        # enter the default sweep / leaderboard (audit hygiene).
        if getattr(pack.meta, "status", "active") == "quarantine":
            skipped.append(
                f"{pack.meta.id} (quarantine: "
                f"{pack.meta.quarantine_reason or 'excluded from default set'})"
            )
            continue
        # Perception sweep: every level × the 4 modality cells
        # (pack:level:<mode>). Overrides both declared configs and the
        # raw enumeration — it is an explicit ablation request.
        if perception_sweep:
            from .scenarios.schema import PERCEPTION_MODES

            unit_iter = []
            for lv in levels:
                for mode in PERCEPTION_MODES:
                    cl = compile_level(pack, lv)
                    cl.fog_mode = mode
                    cl.config_name = f"{lv}:{mode}"
                    unit_iter.append((cl, f"{pack.meta.id}:{lv}:{mode}"))
        # Handoff sweep: each level as base / bad / good handoff cells.
        # `good` needs a winning trajectory from the bank — emitted only
        # when a bank is supplied; `base`/`bad` always run.
        elif handoff_sweep:
            kinds = ["base", "bad"] + (["good"] if handoff_bank else [])
            unit_iter = [
                (compile_level(pack, lv), f"{pack.meta.id}:{lv}:handoff-{kind}")
                for lv in levels
                for kind in kinds
            ]
        # Declared configs (pack:config_name, each pins level+fog_mode)
        # supersede the raw 3-level enumeration when present.
        elif pack.configs:
            from .scenarios.loader import is_map_supported

            ms = is_map_supported(pack.base_map)
            unit_iter = [
                (
                    pack.compile_config(c.name, map_supported=ms),
                    f"{pack.meta.id}:{c.name}",
                )
                for c in pack.configs
            ]
        else:
            # Apply the global fog_mode (from ProviderConfig / CLI) so a
            # single-fog run can audit cells in the `image`/`structured`/
            # `-clear` channels (compiled.fog_mode defaults to vision
            # without this lift, which would silently downgrade every
            # cell to the canonical vision-fogged modality).
            _fog = getattr(provider_cfg, "fog_mode", None) if provider_cfg else None
            unit_iter = []
            for lv in levels:
                cl = compile_level(pack, lv)
                if _fog:
                    cl.fog_mode = _fog
                unit_iter.append((cl, f"{pack.meta.id}:{lv}"))
        for compiled, cell in unit_iter:
            if not compiled.map_supported:
                skipped.append(f"{cell} (map not Rust-loadable)")
                continue
            for split, slist in (("public", seeds), ("held_out", held_out_seeds)):
                for seed in slist:
                    for rep in range(max(1, repeats)):
                        tasks.append((compiled, cell, split, seed, rep))

    def _run_one(task: tuple) -> dict:
        compiled, cell, split, seed, rep = task
        pb = None
        # Only the first repeat writes a Playback — the records (the
        # lightweight per-rep results) carry the pass^k data; saving N
        # full per-turn dumps per cell would just bloat disk.
        if playback_root is not None and rep == 0:
            from .playback import Playback

            pb = Playback(
                Path(playback_root) / f"{run_id}__{_safe_model}",
                f"{cell}:{split}",
                seed,
            )
            pb.run_id, pb.model = run_id, model
        # Audit-format playback (FullPlayback): one JSONL per cell at the
        # canonical `<pack>__<level>__seed<N>__<fog>.jsonl` path the
        # paper-collection script consumes. Same first-repeat gating as
        # the legacy Playback.
        fpb = None
        if full_playback_root is not None and rep == 0:
            from .full_playback import FullPlayback

            # Derive (pack_id, level, fog_mode) from the cell. For
            # perception-sweep cells, the cell is `pack:level:mode`; for
            # legacy/configured cells, fall back to compiled fields.
            parts = cell.split(":")
            _pack_id = compiled.pack_id
            _level = compiled.level
            _fog = getattr(compiled, "fog_mode", "vision") or "vision"
            if len(parts) >= 3:
                _fog = parts[-1]
            # `full_playback_root` is treated as the FINAL per-model dir
            # — callers (e.g. scripts/collect_eval_data.py) already
            # build `<out>/<timestamp>__<model>` and pass it through. We
            # previously appended `<run_id>__<model>` here which
            # produced a double-nested path; if the caller supplied a
            # plain root we still want a per-model subdir, but only if
            # the path doesn't already look like one. Heuristic: if the
            # leaf already starts with the run_id or contains the model
            # safe-name, treat it as final; otherwise append.
            _fp_root = Path(full_playback_root)
            _leaf = _fp_root.name
            if (run_id and _leaf.startswith(run_id)) or _safe_model in _leaf:
                _fp_dir = _fp_root
            else:
                _fp_dir = _fp_root / f"{run_id}__{_safe_model}"
            fpb = FullPlayback(
                _fp_dir,
                pack_id=_pack_id,
                level=_level,
                seed=seed,
                fog_mode=_fog,
            )
        ctrl = factory(compiled)
        if handoff_sweep and ":handoff-" in cell:
            ctrl, _hnote = _handoff_wrap(
                ctrl, cell, seed, handoff_k, handoff_bank
            )
        else:
            _hnote = ""
        res = run_level(compiled, ctrl, seed=seed, playback=pb, full_playback=fpb)
        hstats = getattr(ctrl, "handoff_stats", None)
        if hstats is not None:
            hstats = dict(hstats)
            if _hnote:
                hstats["note"] = _hnote
        sc = score_episode(compiled, res)
        if pb is not None:
            (pb.dir / "score.json").write_text(
                json.dumps(
                    {
                        "composite": sc.composite,
                        "outcome": sc.outcome,
                        "perception": sc.perception,
                        "reasoning": sc.reasoning,
                        "action": sc.action,
                        "weakest_link": sc.weakest_link,
                        # `objective_progress` is deprecated and equals
                        # `objective_blocking_ratio`; kept for one
                        # release of journal back-compat.
                        "objective_progress": res.objective_progress,
                        "objective_blocking_ratio": res.objective_blocking_ratio,
                        "leaves_final": res.leaves_final,
                        "reward_vector": res.reward_vector,
                        "notes": sc.notes,
                    },
                    indent=2,
                )
            )
        return {
            "cell": cell,
            "capability": compiled.meta.capability,
            "split": split,
            "seed": seed,
            "repeat": rep,
            "outcome": sc.outcome,
            "composite": sc.composite,
            "perception": sc.perception,
            "reasoning": sc.reasoning,
            "action": sc.action,
            "weakest_link": sc.weakest_link,
            # `objective_progress` is the deprecated alias of
            # `objective_blocking_ratio`, kept one release for
            # journal back-compat. New consumers should read
            # `leaves_final` directly for per-leaf detail.
            "objective_progress": res.objective_progress,
            "objective_blocking_ratio": res.objective_blocking_ratio,
            "leaves_final": res.leaves_final,
            "reward_vector": res.reward_vector,
            "turns": res.turns,
            "speed": sc.speed,
            "win_turns": sc.win_turns,
            "notes": sc.notes,
            "passivity": hstats.get("passivity") if hstats else None,
            "handoff": hstats,
            "_sc": sc,
        }

    # Pre-flight: dry-run validates compile/selection without engine or
    # API spend; smoke runs exactly one episode.
    if dry_run:
        return {
            "dry_run": True,
            "run_id": run_id,
            "model": model,
            "tasks": len(tasks),
            "skipped": skipped,
            "cells": sorted({t[1] for t in tasks}),
        }
    if smoke:
        tasks = tasks[:1]

    # Checkpoint/resume: a journal of completed episodes. On resume we
    # skip done (pack|level|split|seed|fog_mode) and fold prior records
    # back in, so a killed multi-hour run continues losslessly.
    #
    # Default journal path is deterministic per (out_dir, model) — NOT
    # per-run-id — so a re-launch of `--out <dir> --model <X>` always
    # finds the prior journal and resumes from it. Explicit
    # `--journal-path` overrides.
    jp = journal_path
    if jp is None and playback_root is not None:
        jp = Path(playback_root) / f"_journal__{_safe_model}.jsonl"
    journal = (
        RunJournal(
            jp,
            run_id=run_id,
            model=model,
            code_version=_git_sha(),
            ignore_run_id=ignore_run_id,
        )
        if jp is not None
        else None
    )
    prior: list[dict] = []
    if journal is not None and resume:
        done = journal.done_keys()
        prior = journal.records()

        def _cell_fog(cl):
            """Cell `pack:level:fog` ⇒ fog; otherwise default vision."""
            return getattr(cl, 'fog_mode', None) or 'vision'

        if strict_resume:
            # Strict gate: a key counts as DONE only if (a) the journal
            # has it AND (b) the on-disk score.json exists AND (c) the
            # outcome agrees. Otherwise re-run the cell. This is the
            # production-eval guard that catches the v1.0 sweep's 205
            # journal↔disk mismatches.
            done, prior, _stale_keys = _strict_resume_gate(
                journal, prior, playback_root, run_id, _safe_model,
                progress=progress,
            )
        tasks = [
            t for t in tasks
            if episode_key(t[0].meta.id, t[0].level, t[2], t[3], _cell_fog(t[0])) not in done
        ]

    def _persist(rec: dict) -> None:
        if journal is None:
            return
        slim = {k: v for k, v in rec.items() if k != "_sc"}
        # Cell can be `pack:level` (no fog suffix) or `pack:level:fog`
        # (when perception-sweep / explicit fog_mode is set). The fog
        # suffix matches one of PERCEPTION_MODES.
        cell = rec["cell"]
        parts = cell.split(":")
        from .scenarios.schema import PERCEPTION_MODES
        if len(parts) >= 3 and parts[-1] in PERCEPTION_MODES:
            pack, level, fog = parts[0], parts[1], parts[-1]
        else:
            pack, level = parts[0], parts[1]
            fog = rec.get("fog_mode") or "vision"
        journal.append(
            episode_key(pack, level, rec["split"], rec["seed"], fog),
            slim,
        )

    new_results: list[dict] = []
    truncated = False
    done_n = 0

    def _record(rec: dict) -> None:
        nonlocal done_n
        _persist(rec)
        new_results.append(rec)
        done_n += 1
        if progress is not None:
            progress(done_n, len(tasks), rec, meter.snapshot())
        if report_path is not None:
            # Incremental flush so a long run is always inspectable.
            try:
                write_report(
                    _finalize(prior, new_results, skipped, run_id, model,
                              meter, truncated=False),
                    report_path,
                )
            except Exception:  # noqa: BLE001 — flush must never abort a run
                pass

    try:
        def _safe_run(task: tuple) -> dict:
            # One bad episode (fatal provider 400, engine crash, …) must
            # not abort a multi-hour sweep or lose the report — record
            # it as outcome="error" and continue. Budget is the only
            # signal that intentionally stops the whole run.
            compiled, cell, split, seed, rep = task
            try:
                return _run_one(task)
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                return {
                    "cell": cell,
                    "capability": compiled.meta.capability,
                    "split": split,
                    "seed": seed,
                    "repeat": rep,
                    "outcome": "error",
                    "composite": 0.0,
                    "perception": 0.0,
                    "reasoning": 0.0,
                    "action": 0.0,
                    "weakest_link": "n/a",
                    "objective_progress": 0.0,
                    "objective_blocking_ratio": 0.0,
                    "leaves_final": [],
                    "reward_vector": {},
                    "turns": 0,
                    "notes": [msg[:500]],
                    "_sc": None,
                }

        if concurrency > 1 and len(tasks) > 1:
            if adaptive_concurrency:
                _run_adaptive_pool(
                    tasks, _safe_run, _record, concurrency,
                )
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futs = {ex.submit(_safe_run, t): t for t in tasks}

                    for fu in as_completed(futs):
                        _record(fu.result())
        else:
            for t in tasks:
                _record(_safe_run(t))
    except BudgetExceeded as e:
        truncated = True
        skipped.append(f"BUDGET STOP: {e}")

    out = _finalize(prior, new_results, skipped, run_id, model, meter,
                    truncated=truncated)
    if report_path is not None:
        write_report(out, report_path)
    return out


@dataclass
class _ScoreShim:
    """Reconstruct the fields `_agg` needs from a journaled episode
    dict, so resume aggregates prior + new identically to a fresh run."""

    composite: float
    outcome: str
    perception: float
    reasoning: float
    action: float
    weakest_link: str
    dimensions: dict
    speed: float
    win_turns: int


def _shim(r: dict):
    """Reconstruct a `_ScoreShim` from a journal row.

    For a row written by the live journal-writing path (post commit
    `f9c9c46`) every aggregated field is present. For LEGACY journal
    rows written before that fix, the `speed`/`composite`/`win_turns`/
    perception-reasoning-action subscores can be absent — we fill
    them with 0.0 as a sentinel for "unknown". `_agg` then EXCLUDES
    zero-speed wins from `win_speed_mean` / `win_turns_mean` so the
    sentinel does not pull the win-speed gap toward zero (P1.5 / P1.6
    in PR #30 review). The trinary outcome string is the only legacy
    field we trust unconditionally.
    """
    sc = r.get("_sc")
    if sc is not None:
        return sc
    # Prefer the new `objective_blocking_ratio` scalar; fall back to
    # the deprecated `objective_progress` so a v1.0 journal row still
    # aggregates correctly. Both hold the same min-of-leaves scalar
    # post-fix — only the v1.0 rows carry the old (misleading) mean.
    obj = r.get("objective_blocking_ratio")
    if obj is None:
        obj = r.get("objective_progress", 0.0)
    return _ScoreShim(
        composite=r.get("composite", 0.0),
        outcome=r.get("outcome", "draw"),
        perception=r.get("perception", 0.0),
        reasoning=r.get("reasoning", 0.0),
        action=r.get("action", 0.0),
        weakest_link=r.get("weakest_link", "n/a"),
        dimensions={"objective": obj},
        speed=r.get("speed", 0.0),
        win_turns=r.get("win_turns", r.get("turns", 0)),
    )


def _finalize(prior: list[dict], new: list[dict], skipped: list[str],
              run_id, model, meter, *, truncated: bool) -> dict:
    rows = list(prior) + list(new)
    rows.sort(key=lambda r: (r.get("cell", ""), r.get("split", ""),
                             r.get("seed", 0)))
    by_cell: dict[str, list] = {}
    public_scores: list = []
    held_scores: list = []
    episodes: list[dict] = []
    for r in rows:
        sc = _shim(r)
        slim = {k: v for k, v in r.items() if k != "_sc"}
        if r.get("split") == "public":
            by_cell.setdefault(r["cell"], []).append(sc)
            public_scores.append(sc)
        else:
            held_scores.append(sc)
        episodes.append(slim)

    pub = [r for r in episodes
           if r.get("split") == "public" and r.get("reward_vector")]
    rv_mean: dict = {}
    if pub:
        for k in pub[0]["reward_vector"]:
            rv_mean[k] = round(
                statistics.fmean(r["reward_vector"].get(k, 0.0) for r in pub),
                4,
            )

    out = {
        "run_id": run_id,
        "model": model,
        "truncated": truncated,
        "resumed": len(prior),
        "cost": meter.snapshot() if meter is not None else {},
        "summary": {c: _agg(s) for c, s in by_cell.items()},
        "overall": _agg(public_scores),
        "reward_vector_mean": rv_mean,
        "episodes": episodes,
        "skipped": skipped,
    }
    from .adversarial import adversarial_summary

    adv = adversarial_summary(out)
    if adv["packs"]:
        out["adversarial"] = adv
    if held_scores:
        ho = _agg(held_scores)
        out["overall_held_out"] = ho
        out["generalization_gap"] = round(
            out["overall"].get("composite_mean", 0.0)
            - ho.get("composite_mean", 0.0),
            4,
        )
    return out


def write_report(stats: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


# Family → pack-name prefix map. Keep in sync with the audit CSVs
# (audits/familyN_*.csv). Used by `--family f1,f2,...` to filter the
# bundled `openra_bench/scenarios/packs/` directly — no staging dir
# needed, so `data/packs-*-v3/` etc. are no longer load-bearing.
_FAMILY_PREFIXES = {
    "f1": ("combat-", "action-", "harass-"),
    "f2": ("econ-", "economy-"),
    "f3": ("def-", "defense-", "build-defensive-"),
    "f4": ("scout-", "perception-", "navigation-"),
    "f5": ("lh-", "longhorizon-"),
    "f6": ("build-engineer-", "build-power-", "build-production-",
           "build-rally-", "build-repair-", "build-sell-",
           "build-sequence-", "build-tech-", "building-",
           "tech-", "power-"),
    "f7": ("proc-", "strict-", "maint-", "rob-"),
    "f8": ("mfb-", "mcv-", "coord-", "coordination-"),
    "f9": ("tp-", "tempo-", "strategy-", "adv-", "adversarial-",
           "artofwar-", "risk-", "reasoning-", "expansion-", "mid-"),
    "f10": ("spec-", "custom-"),
    # f10 also covers the rush-hour baseline; matched by exact name below.
}
_F10_EXACT = {"rush-hour"}


def _packs_for_families(family_spec: str) -> list[Path]:
    """Resolve `--family f1,f2,f3to10` (or `--family all`) to the
    matching pack files in the bundled `PACKS_DIR`. Replaces the
    ad-hoc `data/packs-*-v3/` staging dir pattern.

    Family tokens: f1..f10, or compound `f3to10` (= f3+f4+...+f10).
    """
    fams: list[str] = []
    for tok in family_spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok == "all":
            fams.extend(f"f{i}" for i in range(1, 11))
        elif "to" in tok:  # e.g. f3to10
            a, _, b = tok.partition("to")
            ai = int(a.lstrip("f"))
            bi = int(b)
            fams.extend(f"f{i}" for i in range(ai, bi + 1))
        else:
            fams.append(tok)

    seen: set[str] = set()
    prefixes: list[str] = []
    exact: set[str] = set()
    for f in fams:
        if f not in _FAMILY_PREFIXES:
            raise ValueError(f"unknown --family token: {f}")
        for p in _FAMILY_PREFIXES[f]:
            if p not in seen:
                seen.add(p)
                prefixes.append(p)
        if f == "f10":
            exact.update(_F10_EXACT)

    out: list[Path] = []
    for p in sorted(PACKS_DIR.rglob("*.yaml")):
        if p.name.startswith(("_", "TEMPLATE")):
            continue
        stem = p.stem
        if any(stem.startswith(pref) for pref in prefixes) or stem in exact:
            out.append(p)
    return out


def _resolve_packs(spec: str | None,
                   family_spec: str | None = None) -> list[Path]:
    if family_spec:
        return _packs_for_families(family_spec)
    if not spec:
        # Recurse so quarantined packs in `_archive/` are surfaced —
        # they get short-circuited into `skipped` by the quarantine
        # check in `evaluate(...)`, but they MUST be discoverable so
        # the audit hygiene test can confirm the default sweep
        # excludes them.
        return [
            p
            for p in sorted(PACKS_DIR.rglob("*.yaml"))
            if not p.name.startswith(("_", "TEMPLATE"))
        ]
    p = Path(spec)
    return sorted(p.glob("*.yaml")) if p.is_dir() else [p]


def _build_1v1_controller(spec: str | None, compiled: CompiledLevel,
                          side: str, *, provider_cfg=None):
    """Build a Controller (or bare agent_fn) for ONE side of a 1v1
    match. `spec` may be a `scripted:<kind>` literal (escape hatch
    for stall/rusher baselines) or None — None falls back to the
    LLM provider_cfg (if any) via the usual AgentFactory. The
    returned object is passed straight to `run_1v1`."""
    if _is_scripted_spec(spec):
        return _scripted_factory_for_1v1(spec, side)
    # LLM path: identical to the single-player factory, but the
    # ModelAgent gets a side-stamped name so traces are
    # distinguishable.
    if provider_cfg is None:
        # No provider and no scripted spec → fall back to the
        # canonical scripted_explore_agent baseline so the harness
        # still runs (the CLI smoke / scripted-baseline flow).
        return scripted_explore_agent
    from .agent import ModelAgent
    from .game_knowledge import (objective_brief, scenario_primer)
    # Full RA codex + tech tree applied by default (every model sees
    # the same reference); the legacy filtered `unit_codex(codes)`
    # path is no longer wired in.
    return ModelAgent(
        provider_cfg,
        allowed_tools=compiled.scenario.tools,
        objective=objective_brief(
            compiled.scenario.description,
            compiled.win_condition,
            compiled.fail_condition,
            compiled.max_turns,
            getattr(compiled, "objective_coords", "exact"),
        ),
        system_extra=scenario_primer(compiled),
        base_map=compiled.scenario.base_map,
        level=compiled.level,
        fog_mode=getattr(compiled, "fog_mode", "vision"),
        agent_faction=getattr(
            getattr(compiled.scenario, "agent", None), "faction", "") or "",
        enemy_faction=getattr(
            getattr(compiled.scenario, "enemy", None), "faction", "") or "",
    ).agent_fn


def evaluate_1v1(
    packs: list[Path],
    levels: list[str],
    seeds: list[int],
    *,
    provider_cfg=None,
    agent_spec: str | None = None,
    opponent_spec: str | None = "scripted:stall",
    side_swap: bool = False,
    report_path: str | Path | None = None,
    run_id: str | None = None,
    model: str | None = None,
) -> dict:
    """The 1v1 sibling of `evaluate()` — drives `run_1v1` over
    `pack:level:seed` cells and emits an episode-record-compatible
    stats dict. `provider_cfg` runs the agent side; `opponent_spec`
    is a `scripted:<kind>` or `provider:model` opponent (resolved
    identically to provider_cfg via the providers module). When
    `side_swap` is true each match plays TWICE with sides swapped;
    an "ambivalent" outcome (1 win + 1 loss across the two halves)
    is the symmetric draw.

    The stats dict has the same top-level shape as `evaluate()`:
    `run_id`, `model`, `episodes`, `summary` per cell, `overall`,
    and an `adversarial_1v1` block carrying the per-cell win/loss/
    draw breakdown so callers can summarise head-to-head results.
    """
    from .eval_core import _scenario_to_tmp_yaml
    from .one_v_one import run_1v1

    # Opponent provider_cfg: a `provider:model` string is parsed via
    # the same ProviderConfig path the agent uses, so an LLM
    # opponent and an LLM agent share the wire layer.
    opp_provider_cfg = None
    if opponent_spec and not _is_scripted_spec(opponent_spec):
        from .providers import ProviderConfig
        prov, _, model_id = opponent_spec.partition(":")
        opp_provider_cfg = ProviderConfig(
            provider=prov.strip() or "openrouter",
            model=(model_id.strip() or "anthropic/claude-3.5-sonnet"),
        )

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    model = (
        model
        or getattr(provider_cfg, "model", None)
        or "scripted-baseline"
    )
    opponent_label = (
        opponent_spec
        if _is_scripted_spec(opponent_spec)
        else (opponent_spec or "scripted:stall")
    )

    episodes: list[dict] = []
    skipped: list[str] = []
    for pack_path in packs:
        pack = load_pack(pack_path)
        if getattr(pack.meta, "status", "active") == "quarantine":
            skipped.append(f"{pack.meta.id} (quarantine)")
            continue
        if pack.meta.capability != "adversarial":
            skipped.append(
                f"{pack.meta.id} (capability {pack.meta.capability} != "
                f"adversarial — only adversarial packs are valid 1v1 cells)"
            )
            continue
        for lv in levels:
            compiled = compile_level(pack, lv)
            if not compiled.map_supported:
                skipped.append(f"{pack.meta.id}:{lv} (map not Rust-loadable)")
                continue
            cell = f"{pack.meta.id}:{lv}"
            tmp = _scenario_to_tmp_yaml(compiled)
            for seed in seeds:
                halves = ["normal"] + (["swapped"] if side_swap else [])
                outcomes_this_seed: list[str] = []
                for half in halves:
                    if half == "normal":
                        agent_ctrl = _build_1v1_controller(
                            agent_spec, compiled, "agent",
                            provider_cfg=provider_cfg,
                        )
                        enemy_ctrl = _build_1v1_controller(
                            opponent_spec, compiled, "enemy",
                            provider_cfg=opp_provider_cfg,
                        )
                    else:
                        # Swap: agent-slot driven by the opponent,
                        # enemy-slot driven by the agent. Outcome from
                        # the agent's POV is inverted at the end.
                        agent_ctrl = _build_1v1_controller(
                            opponent_spec, compiled, "agent",
                            provider_cfg=opp_provider_cfg,
                        )
                        enemy_ctrl = _build_1v1_controller(
                            agent_spec, compiled, "enemy",
                            provider_cfg=provider_cfg,
                        )
                    res = run_1v1(
                        tmp, agent_ctrl, enemy_ctrl,
                        seed=seed, max_turns=compiled.max_turns,
                    )
                    # Map raw winner ("agent"|"enemy"|"draw") to the
                    # AGENT's POV, accounting for the side swap.
                    if half == "normal":
                        if res.winner == "agent":
                            outcome = "win"
                        elif res.winner == "enemy":
                            outcome = "loss"
                        else:
                            outcome = "draw"
                    else:
                        if res.winner == "enemy":
                            outcome = "win"
                        elif res.winner == "agent":
                            outcome = "loss"
                        else:
                            outcome = "draw"
                    outcomes_this_seed.append(outcome)
                    episodes.append({
                        "cell": cell,
                        "capability": "adversarial",
                        "split": "public",
                        "seed": seed,
                        "side_half": half,
                        "outcome": outcome,
                        "opponent_outcome": (
                            "loss" if outcome == "win"
                            else "win" if outcome == "loss"
                            else "draw"
                        ),
                        "turns": res.turns,
                        "ticks": res.ticks,
                        "reason": res.reason,
                        "agent_name": res.agent_name,
                        "enemy_name": res.enemy_name,
                        "opponent_label": opponent_label,
                        "mode": "1v1",
                    })
                # Side-swap ambivalence: one win + one loss across the
                # two halves is the symmetric DRAW — recorded as a
                # third synthetic record for the cell so callers can
                # see the aggregated head-to-head outcome.
                if side_swap and len(outcomes_this_seed) == 2:
                    wins = sum(1 for o in outcomes_this_seed if o == "win")
                    losses = sum(1 for o in outcomes_this_seed if o == "loss")
                    if wins == 1 and losses == 1:
                        agg = "draw"
                    elif wins == 2:
                        agg = "win"
                    elif losses == 2:
                        agg = "loss"
                    else:
                        agg = "draw"  # any half drew
                    episodes.append({
                        "cell": cell,
                        "capability": "adversarial",
                        "split": "public",
                        "seed": seed,
                        "side_half": "aggregate",
                        "outcome": agg,
                        "turns": 0,
                        "ticks": 0,
                        "reason": "side-swap aggregate",
                        "opponent_label": opponent_label,
                        "mode": "1v1",
                    })

    # Per-cell summary + headline.
    by_cell: dict[str, list[dict]] = {}
    for ep in episodes:
        # Side-swap aggregates are the canonical per-seed outcome when
        # present; otherwise the per-half records each count once.
        by_cell.setdefault(ep["cell"], []).append(ep)
    summary: dict[str, dict] = {}
    for c, eps in by_cell.items():
        # When side-swap aggregates exist for this cell, they are the
        # canonical outcomes (each seed yields one aggregate); when
        # absent, each per-half record counts once.
        canon = [e for e in eps if e.get("side_half") == "aggregate"] or [
            e for e in eps if e.get("side_half") != "aggregate"
        ]
        n = len(canon)
        wins = sum(1 for e in canon if e["outcome"] == "win")
        losses = sum(1 for e in canon if e["outcome"] == "loss")
        draws = sum(1 for e in canon if e["outcome"] == "draw")
        summary[c] = {
            "n": n,
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": round(wins / n, 4) if n else 0.0,
        }

    all_canon = [
        e for e in episodes
        if e.get("side_half") in (None, "aggregate", "normal")
    ]
    if side_swap:
        all_canon = [e for e in episodes if e.get("side_half") == "aggregate"]
    n_all = len(all_canon)
    overall = {
        "n": n_all,
        "wins": sum(1 for e in all_canon if e["outcome"] == "win"),
        "losses": sum(1 for e in all_canon if e["outcome"] == "loss"),
        "draws": sum(1 for e in all_canon if e["outcome"] == "draw"),
        "win_rate": round(
            sum(1 for e in all_canon if e["outcome"] == "win") / n_all, 4
        ) if n_all else 0.0,
    }

    out = {
        "run_id": run_id,
        "model": model,
        "mode": "1v1",
        "opponent": opponent_label,
        "side_swap": bool(side_swap),
        "episodes": episodes,
        "summary": summary,
        "overall": overall,
        "skipped": skipped,
        # Headline block — `adversarial_1v1` is the 1v1-specific
        # roll-up; the existing `adversarial` ladder summary still
        # picks the same packs up when they're scored via `evaluate`.
        "adversarial_1v1": {
            "opponent": opponent_label,
            "win_rate": overall["win_rate"],
            "n_matches": n_all,
            "by_cell": summary,
        },
    }
    if report_path is not None:
        write_report(out, report_path)
    return out


def _status_summary(out_dir: str | Path) -> dict:
    """Read a sweep's playback dir (journal + score.json files) and
    return a status snapshot. Tolerates partial/empty/corrupt journals
    (the production sweeps Ctrl-C frequently; the closer the snapshot
    is to crashes the more torn-line-shaped the journal looks).

    The returned dict is what `run_eval status` prints; tests assert
    on its structure rather than the print-format directly.
    """
    out = {
        "run_dir": str(out_dir),
        "journals": [],
        "header": None,
        "total_journaled": 0,
        "outcomes": {"win": 0, "loss": 0, "draw": 0, "error": 0},
        "compose_mean": 0.0,
        "last_cell": None,
        "errors": [],
        "scores_on_disk": 0,
        "scores_missing": [],
    }
    d = Path(out_dir)
    if not d.exists():
        out["error"] = f"run dir does not exist: {d}"
        return out

    # Find every per-model journal (the deterministic path pattern).
    journals = sorted(d.glob("_journal__*.jsonl"))
    out["journals"] = [str(j) for j in journals]
    rows: list[dict] = []
    header: dict | None = None
    for j in journals:
        try:
            for line in j.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001 — torn tail tolerated
                    continue
                if rec.get("_meta"):
                    if header is None:
                        header = rec
                    continue
                rows.append(rec)
        except OSError as e:  # noqa: PERF203
            out["errors"].append(f"{j.name}: {e}")

    out["header"] = header
    out["total_journaled"] = len(rows)
    comps: list[float] = []
    for r in rows:
        oc = r.get("outcome", "")
        if oc in out["outcomes"]:
            out["outcomes"][oc] += 1
        if isinstance(r.get("composite"), (int, float)):
            comps.append(float(r["composite"]))
    if comps:
        out["compose_mean"] = round(sum(comps) / len(comps), 4)

    # Most recently appended row (journal is append-only, last row in
    # the last journal is the chronologically newest entry).
    if rows:
        last = rows[-1]
        out["last_cell"] = {
            "cell": last.get("cell"),
            "outcome": last.get("outcome"),
            "turns": last.get("turns"),
            "seed": last.get("seed"),
            "composite": last.get("composite"),
        }

    # Cross-check score.json presence (only when journal points at the
    # canonical playback layout — i.e. <run_dir>/<run_id>__<model>/...).
    if header and header.get("run_id"):
        run_id = header.get("run_id")
        # Heuristic: a score.json should exist under a child dir whose
        # name begins with `<run_id>__`.
        per_model = sorted(d.glob(f"{run_id}__*"))
        scored = 0
        for pm in per_model:
            scored += len(list(pm.rglob("score.json")))
        out["scores_on_disk"] = scored
    return out


def _format_status(snap: dict) -> str:
    lines: list[str] = []
    lines.append(f"Run dir:     {snap['run_dir']}")
    h = snap.get("header") or {}
    if h:
        lines.append(
            f"Started:     run_id={h.get('run_id')} "
            f"model={h.get('model')} code={h.get('code_version') or '-'}"
        )
    else:
        lines.append("Started:     (no _meta header — legacy journal)")
    tot = snap["total_journaled"]
    oc = snap["outcomes"]
    won = oc["win"]
    lost = oc["loss"]
    drew = oc["draw"]
    err = oc["error"]
    pct = lambda x, n: f"{round(100 * x / n)}%" if n else "0%"  # noqa: E731
    lines.append(
        f"Cells:       {tot} journaled "
        f"({snap['scores_on_disk']} score.json on disk)"
    )
    lines.append(
        f"Outcomes:    win={won} ({pct(won, tot)}) | "
        f"loss={lost} ({pct(lost, tot)}) | draw={drew} ({pct(drew, tot)}) | "
        f"error={err}"
    )
    lines.append(f"Avg compose: {snap['compose_mean']}")
    last = snap.get("last_cell")
    if last:
        lines.append(
            f"Last cell:   {last['cell']} "
            f"({last['outcome']}, turn {last['turns']}, "
            f"composite {last['composite']})"
        )
    if snap.get("errors"):
        lines.append(f"Errors:      {len(snap['errors'])}")
        for e in snap["errors"][:5]:
            lines.append(f"  • {e}")
    return "\n".join(lines)


def _status_main(argv: list[str]) -> int:
    """Entry for `python -m openra_bench.run_eval status --out <dir>`."""
    ap = argparse.ArgumentParser(
        prog="openra_bench.run_eval status",
        description="Print progress for an in-flight or completed sweep "
                    "by reading the journal(s) + score.json files on disk.",
    )
    ap.add_argument("--out", required=True,
                    help="playback root dir (the one passed as "
                    "--playback to the sweep)")
    a = ap.parse_args(argv)
    snap = _status_summary(a.out)
    print(_format_status(snap))
    return 0


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal, dependency-free .env loader: populate os.environ from
    `KEY=VALUE` lines (skips comments/blanks; never overrides an
    already-set var; strips matching surrounding quotes). Lets
    `--provider openrouter` work straight from a git-ignored .env."""
    import os

    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        k, val = k.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if k and k not in os.environ:
            os.environ[k] = val


def main(argv: list[str]) -> int:
    _load_dotenv()
    # Status subcommand: `python -m openra_bench.run_eval status --out <dir>`.
    # Intentionally a sibling entry point rather than a full
    # argparse-subparsers split — keeps every existing `run_eval ...`
    # invocation working unchanged (back-compat).
    if len(argv) >= 2 and argv[1] == "status":
        return _status_main(argv[2:])
    ap = argparse.ArgumentParser(description="Run a model over OpenRA-Bench scenario packs")
    ap.add_argument("--packs", help="pack file or dir (default: bundled packs/)")
    ap.add_argument(
        "--family",
        help=(
            "Comma-separated family tokens (f1,f2,...,f10,all) or compound "
            "ranges like 'f3to10'. Filters the bundled packs/ dir to the "
            "named families; mutually exclusive with --packs."
        ),
    )
    ap.add_argument("--levels", default="easy,medium,hard")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="run up to N episodes concurrently (each isolated; "
        "report is deterministic regardless)",
    )
    ap.add_argument(
        "--held-out-seeds",
        default="",
        help="comma seeds run as a held-out split; reports the "
        "generalization gap (anti-memorization metric)",
    )
    ap.add_argument("--provider", help="openrouter|vllm|openai|together|bedrock (omit = scripted baseline)")
    ap.add_argument("--model", default="anthropic/claude-3.5-sonnet")
    ap.add_argument("--base-url")
    ap.add_argument(
        "--bedrock-region", default="us-west-2",
        help="AWS region for provider=bedrock. Sonnet 4.6 lives on the "
        "`us.anthropic.claude-sonnet-4-6` cross-region inference profile "
        "served from us-west-2 (default).",
    )
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--out", default="eval_stats.json")
    ap.add_argument(
        "--playback",
        default=None,
        help="dir to save per-episode playback (messages incl. minimap, "
        "per-turn record, manifest, score) so runs can be inspected",
    )
    ap.add_argument(
        "--leaderboard",
        nargs="?",
        const="",
        help="publish this run to the leaderboard store (optional path; "
        "default data/leaderboard.jsonl)",
    )
    # Resilience flags for real OpenRouter runs.
    # 2026-05-24: resume is the DEFAULT. The journal lives at
    # `<playback_root>/_journal__<model>.jsonl` and is shared across
    # re-launches of the same (out_dir, model) so a killed/restarted
    # run automatically picks up where it left off.
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="opt out of automatic resume from the run journal "
                    "(default: resume is ON)")
    ap.add_argument("--resume", dest="resume", action="store_true",
                    default=True,
                    help="skip episodes already in the run journal "
                    "(default behavior)")
    ap.add_argument("--journal", default=None,
                    help="checkpoint journal path (default: under "
                    "<playback>/_journal__<model>.jsonl, deterministic per "
                    "(out_dir, model) so re-launches resume losslessly)")
    ap.add_argument("--strict-resume", action="store_true",
                    help="on resume, verify each journaled cell against "
                    "its on-disk score.json — re-run cells that are "
                    "missing, corrupt, or where the outcome disagrees. "
                    "Recommended for any multi-hour production sweep "
                    "(the v1.0 Qwen 9B sweep had 205/653 journal↔disk "
                    "mismatches without this).")
    ap.add_argument("--ignore-run-id", action="store_true",
                    help="acknowledge a journal whose _meta header "
                    "run_id differs from the current process — i.e. "
                    "explicitly merge two sweep runs into one journal. "
                    "Without this flag a mismatched header aborts.")
    ap.add_argument("--adaptive-concurrency", action="store_true",
                    help="monitor the rolling per-cell error rate and "
                    "halve concurrency when >10% errors over 20 cells, "
                    "restore when <2% over 50 cells. Smooths through "
                    "provider rate-limit storms without aborting.")
    ap.add_argument("--max-spend", type=float, default=0.0,
                    help="hard USD cap; the run finalizes when hit")
    ap.add_argument("--qps", type=float, default=0.0,
                    help="global request/sec throttle (0 = unthrottled)")
    ap.add_argument("--smoke", action="store_true",
                    help="run exactly one episode (live preflight)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate/compile + list tasks, no engine/API")
    ap.add_argument(
        "--or-provider", default="",
        help="OpenRouter: pin a provider/quant endpoint, e.g. "
        "'wandb/bf16' (no fallback) — premium routing off the free pool",
    )
    ap.add_argument("--fog-mode", default="vision",
                    choices=[
                        "vision", "vision-clear",
                        "structured", "structured-clear",
                        "image", "image-clear",
                    ],
                    help="spatial channel: PNG minimap (vision), text fog "
                    "(structured), or image-primary (image). `-clear` "
                    "variants run with no fog of war.")
    ap.add_argument(
        "--full-playback",
        default=None,
        help="audit-format playback dir: one JSONL per cell at "
        "<dir>/<pack>__<level>__seed<N>__<fog>.jsonl with full obs / "
        "request / response / engine warnings. Used by "
        "scripts/collect_eval_data.py for paper-grade data capture.",
    )
    ap.add_argument("--perception-sweep", action="store_true",
                    help="run the 2x2 perception ablation: every "
                    "pack:level expanded into vision/structured x "
                    "fog/no-fog (pack:level:<mode>)")
    ap.add_argument("--handoff-sweep", action="store_true",
                    help="run the handoff ablation: each pack:level as "
                    "handoff-base / handoff-bad (recovery) / handoff-good "
                    "(capitalize) cells")
    ap.add_argument("--handoff-k", type=int, default=3,
                    help="handoff prefix length in turns (default 3)")
    ap.add_argument("--handoff-bank", default=None,
                    help="dir of Playback runs — source of winning "
                    "trajectories for the handoff-good prefix")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each (cell, seed) N times varying only "
                    "model nondeterminism — enables mean +- CI and "
                    "pass^k reliability metrics (needs temperature > 0)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature for the model "
                    "(overrides ProviderConfig.temperature). Set > 0 "
                    "to make --repeats meaningful.")
    # --- 1v1 LLM-vs-LLM mode ---
    # Default is single-player (back-compat). `1v1` routes each
    # (pack, level, seed) through `run_1v1` against the --opponent.
    # Only adversarial-capability packs are valid 1v1 cells; non-
    # adversarial packs are skipped with a reason.
    ap.add_argument(
        "--mode", default="single-player",
        choices=["single-player", "1v1"],
        help="evaluation mode. `single-player` (default) runs the "
        "legacy `run_level` path against scripted bots / scenario "
        "predicates. `1v1` runs each pack:level:seed through "
        "`run_1v1` against the --opponent.",
    )
    ap.add_argument(
        "--opponent", default="scripted:stall",
        help="1v1 opponent spec: `scripted:<kind>` (stall, rusher) "
        "OR `<provider>:<model>` (e.g. openrouter:anthropic/claude-"
        "3.5-sonnet). Ignored in single-player mode.",
    )
    ap.add_argument(
        "--side-swap", action="store_true",
        help="1v1 only: play each match twice with sides swapped, "
        "and emit an aggregate `draw` outcome when one half is won "
        "and the other lost — the symmetric-arena tie-break.",
    )
    a = ap.parse_args(argv[1:])

    cfg = None
    # `scripted:<kind>` is a 1v1-mode escape hatch (no ProviderConfig);
    # it's recognised by `evaluate_1v1` directly so skip the LLM
    # ProviderConfig path entirely for it.
    if a.provider and not _is_scripted_spec(a.provider):
        from .providers import ProviderConfig

        extra_body: dict = {}
        if a.or_provider:
            # OpenRouter routing: `order` takes a provider SLUG;
            # quantization is a separate filter. Accept
            # "provider" or "provider/quant" (e.g. wandb/bf16).
            prov, _, quant = a.or_provider.partition("/")
            pr: dict = {"order": [prov], "allow_fallbacks": False}
            if quant:
                pr["quantizations"] = [quant]
            extra_body["provider"] = pr
        cfg_kw = dict(
            provider=a.provider,
            model=a.model,
            base_url=a.base_url,
            vision=not a.no_vision,
            qps=a.qps,
            fog_mode=a.fog_mode,
            extra_body=extra_body,
        )
        if a.temperature is not None:
            cfg_kw["temperature"] = a.temperature
        if a.provider == "bedrock":
            cfg_kw["bedrock_region"] = a.bedrock_region
        cfg = ProviderConfig(**cfg_kw)

    if a.packs and a.family:
        ap.error("--packs and --family are mutually exclusive")

    # ── 1v1 LLM-vs-LLM branch ────────────────────────────────────────
    # Routes through `evaluate_1v1` which uses `run_1v1` instead of
    # `run_level`. Single-player paths are NOT touched.
    if a.mode == "1v1":
        # The agent side: `--provider scripted:stall` is an escape
        # hatch for the smoke test; otherwise the usual ProviderConfig
        # built above drives the agent. The `agent_spec` carries a
        # `scripted:<kind>` literal through to evaluate_1v1 (its
        # `agent_spec` arg) so the scripted controller for the
        # AGENT side is built identically to the one for the
        # opponent side.
        agent_cfg = cfg
        agent_label = a.model
        agent_spec: str | None = None
        if _is_scripted_spec(a.provider):
            agent_cfg = None
            agent_label = a.provider
            agent_spec = a.provider
        stats = evaluate_1v1(
            _resolve_packs(a.packs, a.family),
            a.levels.split(","),
            [int(s) for s in a.seeds.split(",")],
            provider_cfg=agent_cfg,
            agent_spec=agent_spec,
            opponent_spec=a.opponent,
            side_swap=a.side_swap,
            report_path=a.out,
            model=agent_label,
        )
        write_report(stats, a.out)
        o = stats["overall"]
        print(f"\nwrote {a.out}")
        print(
            f"1v1 overall: n={o.get('n', 0)} win_rate={o.get('win_rate', 0)} "
            f"wins={o.get('wins', 0)} losses={o.get('losses', 0)} "
            f"draws={o.get('draws', 0)} opponent={stats['opponent']}"
        )
        for s in stats["skipped"]:
            print(f"  skipped: {s}")
        return 0

    stats = evaluate(
        _resolve_packs(a.packs, a.family),
        a.levels.split(","),
        [int(s) for s in a.seeds.split(",")],
        provider_cfg=cfg,
        held_out_seeds=[int(s) for s in a.held_out_seeds.split(",") if s.strip()],
        playback_root=a.playback,
        concurrency=a.concurrency,
        model=a.model if a.provider else None,
        journal_path=a.journal,
        resume=a.resume,
        max_spend_usd=a.max_spend,
        smoke=a.smoke,
        dry_run=a.dry_run,
        report_path=a.out,
        perception_sweep=a.perception_sweep,
        handoff_sweep=a.handoff_sweep,
        handoff_k=a.handoff_k,
        handoff_bank=a.handoff_bank,
        repeats=a.repeats,
        full_playback_root=a.full_playback,
        strict_resume=a.strict_resume,
        ignore_run_id=a.ignore_run_id,
        adaptive_concurrency=a.adaptive_concurrency,
        progress=lambda d, n, rec, c: print(
            f"[{d}/{n}] {rec['cell']}:{rec['split']}#{rec['seed']} "
            f"{rec['outcome']} comp={rec['composite']} "
            f"${c['usd']:.4f}", flush=True
        ),
    )
    if stats.get("dry_run"):
        print(f"dry-run: {stats['tasks']} tasks over "
              f"{len(stats['cells'])} cells; skipped {len(stats['skipped'])}")
        return 0
    write_report(stats, a.out)
    o = stats["overall"]
    print(f"\nwrote {a.out}")
    print(
        f"overall: n={o.get('n', 0)} win_rate={o.get('win_rate', 0)} "
        f"composite={o.get('composite_mean', 0)} "
        f"P={o.get('perception_mean', 0)} R={o.get('reasoning_mean', 0)} "
        f"A={o.get('action_mean', 0)} weakest={o.get('weakest_link_hist', {})}"
    )
    if a.leaderboard is not None:
        from .leaderboard import DEFAULT_STORE, ingest_run

        store = a.leaderboard or DEFAULT_STORE
        label = a.model if a.provider else "scripted-baseline"
        rec = ingest_run(stats, label, store)
        print(
            f"published to leaderboard {store}: {label} "
            f"composite={rec['composite']} (episodes={rec['episodes']})"
        )
    for s in stats["skipped"]:
        print(f"  skipped: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
