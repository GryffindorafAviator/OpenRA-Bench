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
import statistics
import sys
from collections import Counter
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

    def factory(compiled: CompiledLevel):
        agent = ModelAgent(
            provider_cfg,
            allowed_tools=compiled.scenario.tools,
            objective=compiled.scenario.description,
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
) -> dict:
    """Run packs×levels×seeds. If `held_out_seeds` is given, those are
    run too and tagged split='held_out'; the report adds
    `overall_held_out` and `generalization_gap` (public composite −
    held-out composite) — the anti-memorization metric the
    generalization literature (Procgen/SMACv2/lmgame-Bench) requires.
    """
    factory = agent_factory or _default_agent_factory(provider_cfg)
    skipped: list[str] = []
    held_out_seeds = held_out_seeds or []

    # Build the flat list of independent episodes (each is fully
    # isolated: own RustEnvPool, own agent, own playback dir) so they
    # can run concurrently.
    tasks: list[tuple] = []
    for pack_path in packs:
        pack = load_pack(pack_path)
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

            pb = Playback(playback_root, f"{cell}:{split}", seed)
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

    if concurrency > 1 and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(_run_one, tasks))
    else:
        results = [_run_one(t) for t in tasks]

    # Deterministic aggregation: sort so the report is identical
    # regardless of worker scheduling.
    results.sort(key=lambda r: (r["cell"], r["split"], r["seed"]))
    by_cell: dict[str, list] = {}
    public_scores: list = []
    held_scores: list = []
    episodes: list[dict] = []
    for r in results:
        sc = r.pop("_sc")
        if r["split"] == "public":
            by_cell.setdefault(r["cell"], []).append(sc)
            public_scores.append(sc)
        else:
            held_scores.append(sc)
        episodes.append(r)

    # Mean cumulative reward vector across public episodes — the
    # scenario-agnostic progress signature, comparable across runs.
    pub = [r for r in episodes if r["split"] == "public" and r.get("reward_vector")]
    rv_mean: dict = {}
    if pub:
        for k in pub[0]["reward_vector"]:
            rv_mean[k] = round(
                statistics.fmean(r["reward_vector"].get(k, 0.0) for r in pub), 4
            )

    out = {
        "summary": {cell: _agg(scs) for cell, scs in by_cell.items()},
        "overall": _agg(public_scores),
        "reward_vector_mean": rv_mean,
        "episodes": episodes,
        "skipped": skipped,
    }
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


def main(argv: list[str]) -> int:
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
    a = ap.parse_args(argv[1:])

    cfg = None
    if a.provider:
        from .providers import ProviderConfig

        cfg = ProviderConfig(
            provider=a.provider,
            model=a.model,
            base_url=a.base_url,
            vision=not a.no_vision,
        )

    stats = evaluate(
        _resolve_packs(a.packs),
        a.levels.split(","),
        [int(s) for s in a.seeds.split(",")],
        provider_cfg=cfg,
        held_out_seeds=[int(s) for s in a.held_out_seeds.split(",") if s.strip()],
        playback_root=a.playback,
        concurrency=a.concurrency,
    )
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
