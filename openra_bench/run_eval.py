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
            ),
            system_extra=scenario_primer(compiled),
            base_map=compiled.scenario.base_map,
            unit_codex=_codex(_scn_codes(compiled)),
            level=compiled.level,
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
        "weakest_link_hist": dict(Counter(s.weakest_link for s in scores)),
    }


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
) -> dict:
    """Run packs×levels×seeds. If `held_out_seeds` is given, those are
    run too and tagged split='held_out'; the report adds
    `overall_held_out` and `generalization_gap` (public composite −
    held-out composite) — the anti-memorization metric the
    generalization literature (Procgen/SMACv2/lmgame-Bench) requires.
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
                ),
                provider=shared,
                system_extra=scenario_primer(compiled),
                base_map=compiled.scenario.base_map,
                unit_codex=_codex(_scn_codes(compiled)),
                level=compiled.level,
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
        for level in levels:
            compiled = compile_level(pack, level)
            if not compiled.map_supported:
                skipped.append(f"{pack.meta.id}:{level} (map not Rust-loadable)")
                continue
            cell = f"{pack.meta.id}:{level}"
            for split, slist in (("public", seeds), ("held_out", held_out_seeds)):
                for seed in slist:
                    tasks.append((compiled, cell, split, seed))

    def _run_one(task: tuple) -> dict:
        compiled, cell, split, seed = task
        pb = None
        if playback_root is not None:
            from .playback import Playback

            pb = Playback(
                Path(playback_root) / f"{run_id}__{_safe_model}",
                f"{cell}:{split}",
                seed,
            )
            pb.run_id, pb.model = run_id, model
        res = run_level(compiled, factory(compiled), seed=seed, playback=pb)
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
            compiled, cell, split, seed = task
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
        return [
            p
            for p in sorted(PACKS_DIR.glob("*.yaml"))
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
    ap.add_argument("--provider", help="openrouter|vllm|openai (omit = scripted baseline)")
    ap.add_argument("--model", default="anthropic/claude-3.5-sonnet")
    ap.add_argument("--base-url")
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
    a = ap.parse_args(argv[1:])

    cfg = None
    if a.provider:
        from .providers import ProviderConfig

        cfg = ProviderConfig(
            provider=a.provider,
            model=a.model,
            base_url=a.base_url,
            vision=not a.no_vision,
            qps=a.qps,
        )

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
