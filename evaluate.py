#!/usr/bin/env python3
"""OpenRA-Bench evaluation harness.

Runs N games of an agent against a built-in AI opponent, collects metrics,
and appends aggregate results to data/results.csv.

Usage:
    # Option A: Local server (Docker)
    docker compose up openra-rl
    python evaluate.py \
        --agent scripted \
        --agent-name "ScriptedBot-v1" \
        --opponent hard \
        --games 10 \
        --server http://localhost:8000

    # Option B: HuggingFace-hosted server (no Docker needed)
    python evaluate.py \
        --agent scripted \
        --agent-name "ScriptedBot-v1" \
        --opponent hard \
        --games 10 \
        --server https://openra-rl-openra-rl.hf.space

    # Dry run (validate args without connecting):
    python evaluate.py --dry-run --agent-name "Test" --games 5
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen

from openra_rl_util.rubrics import compute_composite_score_from_games, compute_game_metrics

# Evaluation results file
RESULTS_FILE = Path(__file__).parent / "data" / "results.csv"

RESULTS_COLUMNS = [
    "agent_name",
    "agent_type",
    "opponent",
    "games",
    "win_rate",
    "score",
    "avg_kills",
    "avg_deaths",
    "kd_ratio",
    "avg_economy",
    "avg_game_length",
    "timestamp",
    "replay_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenRA-Bench: Evaluate agents against AI opponents"
    )
    parser.add_argument(
        "--agent",
        choices=["scripted", "llm", "mcp", "custom"],
        default="scripted",
        help="Agent type to run (default: scripted)",
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        help="Name for this agent on the leaderboard",
    )
    parser.add_argument(
        "--agent-type",
        choices=["Scripted", "LLM", "RL"],
        help="Leaderboard category (auto-detected from --agent if not set)",
    )
    parser.add_argument(
        "--opponent",
        choices=["Easy", "Normal", "Hard"],
        default="Normal",
        help="AI opponent difficulty (default: Normal)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to play (default: 10)",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="OpenRA-RL server URL. Use http://localhost:8000 for local Docker, "
        "or https://openra-rl-openra-rl.hf.space for HuggingFace-hosted",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Max steps per game before timeout (default: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and show what would run, without connecting",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_FILE,
        help=f"Output CSV path (default: {RESULTS_FILE})",
    )
    args = parser.parse_args()

    # Auto-detect agent type
    if args.agent_type is None:
        type_map = {"scripted": "Scripted", "llm": "LLM", "mcp": "Scripted", "custom": "RL"}
        args.agent_type = type_map[args.agent]

    return args


async def run_game(env: Any, agent_fn: Any, max_steps: int) -> Dict[str, Any]:
    """Run a single game and return metrics.

    Args:
        env: OpenRAEnv client instance.
        agent_fn: Callable(obs) -> action.
        max_steps: Maximum steps before timeout.

    Returns:
        Dict with game metrics (from compute_game_metrics).
    """
    obs = await env.reset()
    steps = 0

    while not obs.done and steps < max_steps:
        action = agent_fn(obs)
        obs = await env.step(action)
        steps += 1

    return compute_game_metrics(obs)


def get_agent_fn(agent_type: str) -> Any:
    """Get the agent decision function for the specified type.

    Returns a callable that takes an observation and returns an action.
    """
    if agent_type == "scripted":
        from openra_env.models import OpenRAAction
        # Simple no-op agent for evaluation framework testing
        # TODO: Wire to openra_env.agents.scripted.ScriptedBot when extracted
        return lambda obs: OpenRAAction(commands=[])
    else:
        from openra_env.models import OpenRAAction
        return lambda obs: OpenRAAction(commands=[])


def _wake_hf_space(server_url: str, max_wait: int = 120) -> None:
    """Send HTTP request to wake a sleeping HuggingFace Space.

    HF Spaces sleep after inactivity. An HTTP GET wakes them up,
    but it may take up to ~2 minutes for the container to start.
    """
    if ".hf.space" not in server_url:
        return

    print(f"  Waking HuggingFace Space...", end=" ", flush=True)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            with urlopen(server_url, timeout=10) as resp:
                if resp.status == 200:
                    print("ready.")
                    return
        except Exception:
            pass
        time.sleep(5)
    print("timed out (Space may still be starting).")


async def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full evaluation: N games, collect metrics, compute aggregates."""
    from openra_env.client import OpenRAEnv

    _wake_hf_space(args.server)

    agent_fn = get_agent_fn(args.agent)
    game_results: List[Dict[str, Any]] = []

    async with OpenRAEnv(args.server) as env:
        for i in range(args.games):
            print(f"  Game {i + 1}/{args.games}...", end=" ", flush=True)
            metrics = await run_game(env, agent_fn, args.max_steps)
            game_results.append(metrics)
            result_str = metrics["result"] or "timeout"
            print(f"{result_str} (ticks: {metrics['ticks']}, K/D: {metrics['kd_ratio']:.1f})")

    # Aggregate results using single source of truth from openra-rl-util
    wins = sum(1 for g in game_results if g["win"])
    total = len(game_results)

    return {
        "agent_name": args.agent_name,
        "agent_type": args.agent_type,
        "opponent": args.opponent,
        "games": total,
        "win_rate": round(100.0 * wins / max(total, 1), 1),
        "score": round(compute_composite_score_from_games(game_results), 1),
        "avg_kills": round(sum(g["kills_cost"] for g in game_results) / max(total, 1)),
        "avg_deaths": round(sum(g["deaths_cost"] for g in game_results) / max(total, 1)),
        "kd_ratio": round(
            sum(g["kd_ratio"] for g in game_results) / max(total, 1), 2
        ),
        "avg_economy": round(
            sum(g["assets_value"] for g in game_results) / max(total, 1)
        ),
        "avg_game_length": round(
            sum(g["ticks"] for g in game_results) / max(total, 1)
        ),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "replay_url": "",
    }


def append_results(results: Dict[str, Any], output_path: Path) -> None:
    """Append evaluation results to CSV file."""
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


def main() -> None:
    args = parse_args()

    print(f"OpenRA-Bench Evaluation")
    print(f"  Agent: {args.agent_name} ({args.agent_type})")
    print(f"  Opponent: {args.opponent}")
    print(f"  Games: {args.games}")
    print(f"  Server: {args.server}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would run evaluation with the above settings.")
        print(f"[DRY RUN] Results would be written to: {args.output}")
        return

    results = asyncio.run(run_evaluation(args))

    print()
    print(f"Results:")
    print(f"  Win Rate: {results['win_rate']}%")
    print(f"  Score: {results['score']}")
    print(f"  K/D Ratio: {results['kd_ratio']}")
    print(f"  Avg Economy: {results['avg_economy']}")
    print(f"  Avg Game Length: {results['avg_game_length']} ticks")

    append_results(results, args.output)
    print(f"\nResults appended to {args.output}")


if __name__ == "__main__":
    main()
