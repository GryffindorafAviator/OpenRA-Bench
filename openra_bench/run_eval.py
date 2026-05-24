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
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .eval_core import run_level, scripted_explore_agent
from .scenarios import load_pack
from .scenarios.loader import PACKS_DIR, compile_level
from .scenarios.schema import CompiledLevel
from .scoring import score_episode

# agent_factory: (CompiledLevel) -> agent_fn(render_state, Command)->[Command]
AgentFactory = Callable[[CompiledLevel], Callable]


def _default_agent_factory(provider_cfg) -> AgentFactory:
    if provider_cfg is None:
        return lambda _c: scripted_explore_agent
    from .agent import ModelAgent

    from .game_knowledge import (actor_codes, objective_brief,
                                 scenario_primer)
    from .prompt_v2 import unit_codex as _codex
    def _scn_codes(c):
        from .game_knowledge import _condition_codes
        return (actor_codes(c.scenario) | _condition_codes(c.win_condition)
                | _condition_codes(c.fail_condition))

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
            unit_codex=_codex(_scn_codes(compiled)),
            level=compiled.level,
            fog_mode=getattr(compiled, "fog_mode", "vision"),
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
        "win_speed_mean": round(
            statistics.fmean([s.speed for s in scores if s.outcome == "win"]), 4
        ) if any(s.outcome == "win" for s in scores) else 0.0,
        "win_turns_mean": round(
            statistics.fmean(
                [s.win_turns for s in scores if s.outcome == "win"]
            ), 2
        ) if any(s.outcome == "win" for s in scores) else 0.0,
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
    resume: bool = False,
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

        from .game_knowledge import (actor_codes, objective_brief,
                                     scenario_primer)
        from .prompt_v2 import unit_codex as _codex
        def _scn_codes(c):
            from .game_knowledge import _condition_codes
            return (actor_codes(c.scenario) | _condition_codes(c.win_condition)
                    | _condition_codes(c.fail_condition))

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
                unit_codex=_codex(_scn_codes(compiled)),
                level=compiled.level,
                fog_mode=getattr(compiled, "fog_mode", "vision"),
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
                        "objective_progress": res.objective_progress,
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
            "objective_progress": res.objective_progress,
            "reward_vector": res.reward_vector,
            "turns": res.turns,
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
    # skip done (pack|level|split|seed) and fold prior records back in,
    # so a killed multi-hour run continues losslessly.
    jp = journal_path
    if jp is None and playback_root is not None:
        jp = Path(playback_root) / f"{run_id}__{_safe_model}" / "_journal.jsonl"
    journal = RunJournal(jp) if jp is not None else None
    prior: list[dict] = []
    if journal is not None and resume:
        done = journal.done_keys()
        prior = journal.records()
        tasks = [
            t for t in tasks
            if episode_key(t[0].meta.id, t[0].level, t[2], t[3]) not in done
        ]

    def _persist(rec: dict) -> None:
        if journal is None:
            return
        slim = {k: v for k, v in rec.items() if k != "_sc"}
        journal.append(
            episode_key(
                rec["cell"].rsplit(":", 1)[0],
                rec["cell"].rsplit(":", 1)[1],
                rec["split"],
                rec["seed"],
            ),
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
                    "reward_vector": {},
                    "turns": 0,
                    "notes": [msg[:500]],
                    "_sc": None,
                }

        if concurrency > 1 and len(tasks) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(_safe_run, t): t for t in tasks}
                from concurrent.futures import as_completed

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


def _shim(r: dict):
    sc = r.get("_sc")
    if sc is not None:
        return sc
    return _ScoreShim(
        composite=r.get("composite", 0.0),
        outcome=r.get("outcome", "draw"),
        perception=r.get("perception", 0.0),
        reasoning=r.get("reasoning", 0.0),
        action=r.get("action", 0.0),
        weakest_link=r.get("weakest_link", "n/a"),
        dimensions={"objective": r.get("objective_progress", 0.0)},
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


def _resolve_packs(spec: str | None) -> list[Path]:
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
    ap = argparse.ArgumentParser(description="Run a model over OpenRA-Bench scenario packs")
    ap.add_argument("--packs", help="pack file or dir (default: bundled packs/)")
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
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes already in the run journal")
    ap.add_argument("--journal", default=None,
                    help="checkpoint journal path (default: under --playback)")
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
    a = ap.parse_args(argv[1:])

    cfg = None
    if a.provider:
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

    stats = evaluate(
        _resolve_packs(a.packs),
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
