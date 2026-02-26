"""In-browser evaluation runner for OpenRA-Bench.

Runs games against the OpenRA-RL server via HTTP REST API (POST /reset, /step).
No openra-rl/openenv imports — avoids websockets version conflicts with Gradio.

Scoring logic inlined from openra_rl_util.rubrics (which has zero dependencies).
"""

import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import httpx

# HuggingFace-hosted OpenRA-RL environment
DEFAULT_SERVER = "https://openra-rl-openra-rl.hf.space"
MAX_STEPS_PER_GAME = 5000
STEP_TIMEOUT = 60.0


# ── Scoring (inlined from openra_rl_util/rubrics.py) ────────────────────────


def compute_game_metrics(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract benchmark metrics from a final game observation dict."""
    military = obs.get("military") or {}
    economy = obs.get("economy") or {}

    kills = military.get("kills_cost", 0)
    deaths = military.get("deaths_cost", 0)
    assets = military.get("assets_value", 0)
    cash = economy.get("cash", 0)
    result = obs.get("result", "")
    tick = obs.get("tick", 0)

    return {
        "result": result,
        "win": result == "win",
        "ticks": tick,
        "kills_cost": kills,
        "deaths_cost": deaths,
        "kd_ratio": kills / max(deaths, 1),
        "assets_value": assets,
        "cash": cash,
    }


def compute_composite_score(game_results: List[Dict[str, Any]]) -> float:
    """Compute OpenRA-Bench composite score: 50% win + 25% military + 25% economy."""
    total = len(game_results)
    if total == 0:
        return 0.0

    win_rate = sum(1 for g in game_results if g["win"]) / total

    mil_scores = []
    for g in game_results:
        kills, deaths = g["kills_cost"], g["deaths_cost"]
        total_cost = kills + deaths
        mil_scores.append(kills / total_cost if total_cost > 0 else 0.5)
    avg_mil = sum(mil_scores) / total

    econ_scores = []
    for g in game_results:
        assets = g["assets_value"]
        econ_scores.append(assets / (assets + 10000) if assets >= 0 else 0.0)
    avg_econ = sum(econ_scores) / total

    return 100.0 * (0.5 * win_rate + 0.25 * avg_mil + 0.25 * avg_econ)


# ── Server communication ────────────────────────────────────────────────────


def wake_hf_space(server_url: str, max_wait: int = 120) -> str:
    """Wake a sleeping HuggingFace Space. Returns status message."""
    if ".hf.space" not in server_url:
        return "Local server, skipping wake."

    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = httpx.get(server_url, timeout=10, follow_redirects=True)
            if resp.status_code == 200:
                return "Environment server is ready."
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return "Warning: server may still be starting."


async def run_single_game(
    client: httpx.AsyncClient,
    server_url: str,
    max_steps: int = MAX_STEPS_PER_GAME,
) -> Dict[str, Any]:
    """Run one game via HTTP REST and return metrics."""
    # Reset environment
    resp = await client.post(f"{server_url}/reset", json={})
    resp.raise_for_status()
    data = resp.json()
    obs = data["observation"]

    steps = 0
    while not obs.get("result") and steps < max_steps:
        # Scripted no-op agent: send empty commands
        action = {"commands": []}
        resp = await client.post(
            f"{server_url}/step",
            json={"action": action},
        )
        resp.raise_for_status()
        data = resp.json()
        obs = data["observation"]
        steps += 1

    return compute_game_metrics(obs)


async def run_evaluation(
    agent_name: str,
    opponent: str,
    num_games: int,
    server_url: str = DEFAULT_SERVER,
    on_game_done: Optional[Callable[[int, int, Dict], None]] = None,
) -> Dict[str, Any]:
    """Run N games and return aggregate results.

    Args:
        agent_name: Display name for the leaderboard.
        opponent: AI difficulty (Beginner/Easy/Medium/Normal/Hard).
        num_games: Number of games to play.
        server_url: OpenRA-RL server URL.
        on_game_done: Optional callback(game_num, total, metrics) after each game.

    Returns:
        Dict with all fields needed for results.csv.
    """
    game_results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=STEP_TIMEOUT) as client:
        for i in range(num_games):
            metrics = await run_single_game(client, server_url)
            game_results.append(metrics)
            if on_game_done:
                on_game_done(i + 1, num_games, metrics)

    total = len(game_results)
    wins = sum(1 for g in game_results if g["win"])

    return {
        "agent_name": agent_name,
        "agent_type": "Scripted",
        "opponent": opponent,
        "games": total,
        "win_rate": round(100.0 * wins / max(total, 1), 1),
        "score": round(compute_composite_score(game_results), 1),
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
