"""OpenRA-Bench: Agent Leaderboard for OpenRA-RL.

A Gradio app that displays agent rankings, supports filtering by type
and opponent difficulty, and lets users run evaluations in-browser.

Run locally:
    python app.py

Deploy on HuggingFace Spaces:
    Push app.py, requirements.txt, data/, and README.md to your HF Space.
"""

import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from evaluate_runner import DEFAULT_SERVER, run_evaluation, wake_hf_space

# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "results.csv"

AGENT_TYPE_COLORS = {
    "Scripted": "#ffcd75",  # Gold
    "LLM": "#7497db",       # Blue
    "RL": "#75809c",        # Gray-blue
}

DISPLAY_COLUMNS = [
    "Rank",
    "Agent",
    "Type",
    "Opponent",
    "Games",
    "Win Rate (%)",
    "Score",
    "K/D Ratio",
    "Avg Kills",
    "Avg Deaths",
    "Avg Economy",
    "Avg Game Length",
    "Date",
]


def load_data() -> pd.DataFrame:
    """Load leaderboard data from CSV."""
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Rename for display
    df = df.rename(columns={
        "agent_name": "Agent",
        "agent_type": "Type",
        "opponent": "Opponent",
        "games": "Games",
        "win_rate": "Win Rate (%)",
        "score": "Score",
        "kd_ratio": "K/D Ratio",
        "avg_kills": "Avg Kills",
        "avg_deaths": "Avg Deaths",
        "avg_economy": "Avg Economy",
        "avg_game_length": "Avg Game Length",
        "timestamp": "Date",
    })

    return df[DISPLAY_COLUMNS]


def add_type_badges(df: pd.DataFrame) -> pd.DataFrame:
    """Add color-coded HTML badges to the Type column."""
    def badge(agent_type: str) -> str:
        color = AGENT_TYPE_COLORS.get(agent_type, "#ccc")
        text_color = "#fff" if agent_type != "Scripted" else "#333"
        return (
            f'<span style="background:{color};color:{text_color};'
            f'padding:2px 8px;border-radius:4px;font-size:0.85em">'
            f"{agent_type}</span>"
        )

    df = df.copy()
    df["Type"] = df["Type"].apply(badge)
    return df


# ── Filtering ─────────────────────────────────────────────────────────────────


def filter_leaderboard(
    search: str,
    agent_types: list[str],
    opponent: str,
) -> pd.DataFrame:
    """Filter leaderboard by search, agent type, and opponent."""
    df = load_data()

    # Filter by agent type
    if agent_types:
        df = df[df["Type"].isin(agent_types)]

    # Filter by opponent
    if opponent and opponent != "All":
        df = df[df["Opponent"] == opponent]

    # Search by agent name (regex)
    if search and search.strip():
        patterns = [p.strip() for p in search.split(",") if p.strip()]
        mask = pd.Series([False] * len(df), index=df.index)
        for pattern in patterns:
            mask |= df["Agent"].str.contains(pattern, case=False, regex=True, na=False)
        df = df[mask]

    # Re-rank after filtering
    df = df.reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    return add_type_badges(df)


# ── Result Persistence ────────────────────────────────────────────────────────

SUBMISSIONS_DIR = Path(__file__).parent / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# CommitScheduler pushes submissions to HF dataset (only on HF Spaces)
_scheduler = None
if os.environ.get("HF_TOKEN") and os.environ.get("SPACE_ID"):
    try:
        from huggingface_hub import CommitScheduler

        _scheduler = CommitScheduler(
            repo_id="openra-rl/bench-results",
            repo_type="dataset",
            folder_path=str(SUBMISSIONS_DIR),
            every=5,
            token=os.environ["HF_TOKEN"],
        )
    except Exception:
        pass  # Running locally without HF token — skip


def save_submission(results: dict) -> None:
    """Append results to local JSONL and CSV."""
    # JSONL for CommitScheduler → HF dataset
    jsonl_path = SUBMISSIONS_DIR / "results.jsonl"
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    # Also append to data/results.csv for the leaderboard
    csv_path = DATA_PATH
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    fieldnames = [
        "agent_name", "agent_type", "opponent", "games", "win_rate",
        "score", "avg_kills", "avg_deaths", "kd_ratio", "avg_economy",
        "avg_game_length", "timestamp", "replay_url",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


# ── Evaluation Handler ────────────────────────────────────────────────────────


def run_eval_sync(agent_name: str, opponent: str, num_games: int):
    """Generator that runs evaluation and yields progress updates."""
    if not agent_name or not agent_name.strip():
        yield "Error: Please enter an agent name.", None, ""
        return

    agent_name = agent_name.strip()
    num_games = int(num_games)

    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Wake server
    yield log(f"Connecting to {DEFAULT_SERVER}..."), None, ""
    status = wake_hf_space(DEFAULT_SERVER)
    yield log(status), None, ""

    # Track per-game progress
    game_log = []

    def on_game_done(game_num, total, metrics):
        result = metrics["result"] or "timeout"
        kd = metrics["kd_ratio"]
        game_log.append({
            "Game": game_num,
            "Result": result,
            "K/D": round(kd, 1),
            "Ticks": metrics["ticks"],
        })

    yield log(f"Running {num_games} game(s) vs {opponent} AI..."), None, ""

    try:
        results = asyncio.run(
            run_evaluation(
                agent_name=agent_name,
                opponent=opponent,
                num_games=num_games,
                server_url=DEFAULT_SERVER,
                on_game_done=on_game_done,
            )
        )
    except Exception as e:
        yield log(f"Error: {e}"), None, ""
        return

    # Save results
    save_submission(results)

    # Format output
    for g in game_log:
        log(f"  Game {g['Game']}: {g['Result']} (K/D: {g['K/D']}, ticks: {g['Ticks']})")

    log(f"\nEvaluation complete!")

    summary = (
        f"### Results: {agent_name}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| **Score** | **{results['score']}** |\n"
        f"| Win Rate | {results['win_rate']}% |\n"
        f"| K/D Ratio | {results['kd_ratio']} |\n"
        f"| Avg Economy | {results['avg_economy']} |\n"
        f"| Games | {results['games']} vs {results['opponent']} |\n"
    )

    results_df = pd.DataFrame([{
        "Agent": results["agent_name"],
        "Type": results["agent_type"],
        "Opponent": results["opponent"],
        "Games": results["games"],
        "Win Rate (%)": results["win_rate"],
        "Score": results["score"],
        "K/D Ratio": results["kd_ratio"],
    }])

    yield "\n".join(log_lines), results_df, summary


# ── UI ────────────────────────────────────────────────────────────────────────

ABOUT_MD = """
## What is OpenRA-Bench?

**OpenRA-Bench** is a standardized benchmark for evaluating AI agents playing
[Red Alert](https://www.openra.net/) through the
[OpenRA-RL](https://openra-rl.dev) environment.

### Evaluation Protocol

- **Game**: Red Alert (OpenRA engine)
- **Format**: 1v1 agent vs built-in AI
- **Opponents**: Easy, Normal, Hard difficulty
- **Games per entry**: Minimum 10 games per configuration
- **Metrics**: Win rate, composite score, K/D ratio, economy

### Composite Score

The benchmark score combines three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Win Rate | 50% | Percentage of games won |
| Military Efficiency | 25% | Kill/death cost ratio (normalized) |
| Economy | 25% | Final asset value (normalized) |

### Agent Types

- **Scripted**: Rule-based bots with hardcoded strategies
- **LLM**: Language model agents (Claude, GPT, etc.)
- **RL**: Reinforcement learning policies (PPO, SAC, etc.)

### Links

- [OpenRA-RL Documentation](https://openra-rl.dev)
- [GitHub Repository](https://github.com/yxc20089/OpenRA-RL)
- [OpenRA-Bench Source](https://github.com/yxc20089/OpenRA-Bench)
- [OpenEnv Framework](https://huggingface.co/openenv)
- [HuggingFace Space](https://huggingface.co/spaces/openra-rl/OpenRA-Bench)
"""

SUBMIT_MD = """
## How to Submit Results

### Option A: In-Browser (no setup needed)

Use the **Evaluate** tab to run a scripted agent directly from your browser.
Results are saved to the leaderboard automatically.

### Option B: CLI with HuggingFace-hosted server (no Docker needed)

```bash
git clone https://github.com/yxc20089/OpenRA-Bench.git
cd OpenRA-Bench
pip install -r requirements.txt
pip install openra-rl openra-rl-util

python evaluate.py \\
    --agent scripted \\
    --agent-name "MyBot-v1" \\
    --agent-type Scripted \\
    --opponent Normal \\
    --games 10 \\
    --server https://openra-rl-openra-rl.hf.space
```

### Option C: Local server (Docker)**

```bash
git clone --recursive https://github.com/yxc20089/OpenRA-RL.git
cd OpenRA-RL && pip install -e . && docker compose up openra-rl
cd /path/to/OpenRA-Bench

python evaluate.py \\
    --agent scripted \\
    --agent-name "MyBot-v1" \\
    --agent-type Scripted \\
    --opponent Normal \\
    --games 10 \\
    --server http://localhost:8000
```

### 3. Submit via Pull Request

1. Fork [OpenRA-Bench](https://github.com/yxc20089/OpenRA-Bench)
2. Run the evaluation (results append to `data/results.csv`)
3. Commit and open a PR with:
   - Your updated `data/results.csv`
   - A description of your agent
   - (Optional) Replay files in `replays/`

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `--agent` | Agent type: `scripted`, `llm`, `mcp`, `custom` |
| `--agent-name` | Display name on the leaderboard |
| `--agent-type` | Category: `Scripted`, `LLM`, `RL` |
| `--opponent` | AI difficulty: `Easy`, `Normal`, `Hard` |
| `--games` | Number of games (minimum 10) |
| `--server` | OpenRA-RL server URL (local or HuggingFace-hosted) |

### Custom Agents

For custom agents, implement the standard `reset/step` loop:

```python
from openra_env.client import OpenRAEnv
from openra_env.models import OpenRAAction

async with OpenRAEnv("https://openra-rl-openra-rl.hf.space") as env:
    obs = await env.reset()
    while not obs.done:
        action = your_agent.decide(obs)
        obs = await env.step(action)
```

Then run `evaluate.py --agent custom` with your agent integrated.
"""


def build_app() -> gr.Blocks:
    """Build the Gradio leaderboard app."""
    initial_df = add_type_badges(load_data())

    with gr.Blocks(title="OpenRA-Bench") as app:
        gr.Markdown(
            "# OpenRA-Bench\n"
            "**Agent Leaderboard for OpenRA-RL** — "
            "Train AI to Play Real-Time Strategy"
        )

        with gr.Tabs():
            # ── Leaderboard Tab ───────────────────────────────────────────
            with gr.Tab("Leaderboard"):
                with gr.Row():
                    search_box = gr.Textbox(
                        label="Search agents",
                        placeholder="Search by name (supports regex, comma-separated)...",
                        scale=3,
                    )
                    type_filter = gr.CheckboxGroup(
                        choices=["Scripted", "LLM", "RL"],
                        value=["Scripted", "LLM", "RL"],
                        label="Agent Type",
                        scale=2,
                    )
                    opponent_filter = gr.Dropdown(
                        choices=["All", "Easy", "Normal", "Hard"],
                        value="All",
                        label="Opponent",
                        scale=1,
                    )

                leaderboard = gr.Dataframe(
                    value=initial_df,
                    datatype=[
                        "number",    # Rank
                        "str",       # Agent
                        "html",      # Type (badge)
                        "str",       # Opponent
                        "number",    # Games
                        "number",    # Win Rate
                        "number",    # Score
                        "number",    # K/D Ratio
                        "number",    # Avg Kills
                        "number",    # Avg Deaths
                        "number",    # Avg Economy
                        "number",    # Avg Game Length
                        "str",       # Date
                    ],
                    interactive=False,
                    show_label=False,
                )

                # Wire up filters
                for component in [search_box, type_filter, opponent_filter]:
                    component.change(
                        fn=filter_leaderboard,
                        inputs=[search_box, type_filter, opponent_filter],
                        outputs=leaderboard,
                    )

            # ── Evaluate Tab ──────────────────────────────────────────────
            with gr.Tab("Evaluate"):
                gr.Markdown(
                    "## Run Evaluation\n\n"
                    "Run a scripted agent against the HuggingFace-hosted "
                    "OpenRA-RL environment. No Docker or local setup needed."
                )
                with gr.Row():
                    eval_name = gr.Textbox(
                        label="Agent Name",
                        placeholder="e.g. MyBot-v1",
                        scale=2,
                    )
                    eval_opponent = gr.Dropdown(
                        choices=["Easy", "Normal", "Hard"],
                        value="Normal",
                        label="Opponent",
                        scale=1,
                    )
                    eval_games = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=3,
                        step=1,
                        label="Games",
                        scale=1,
                    )
                eval_btn = gr.Button("Run Evaluation", variant="primary")

                eval_log = gr.Textbox(
                    label="Progress",
                    lines=10,
                    interactive=False,
                )
                eval_results = gr.Dataframe(
                    label="Game Results",
                    interactive=False,
                )
                eval_summary = gr.Markdown()

                eval_btn.click(
                    fn=run_eval_sync,
                    inputs=[eval_name, eval_opponent, eval_games],
                    outputs=[eval_log, eval_results, eval_summary],
                )

            # ── About Tab ─────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)

            # ── Submit Tab ────────────────────────────────────────────────
            with gr.Tab("Submit"):
                gr.Markdown(SUBMIT_MD)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
