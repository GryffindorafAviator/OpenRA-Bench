"""OpenRA-Bench: Agent Leaderboard for OpenRA-RL.

A Gradio app that displays agent rankings, supports filtering by type
and opponent difficulty, and provides submission instructions.

Run locally:
    python app.py

Deploy on HuggingFace Spaces:
    Push app.py, requirements.txt, data/, and README.md to your HF Space.
"""

import os
from pathlib import Path

import gradio as gr
import pandas as pd

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

### 1. Set up the evaluation harness

```bash
git clone https://github.com/yxc20089/OpenRA-Bench.git
cd OpenRA-Bench
pip install -r requirements.txt
pip install openra-rl openra-rl-util
```

### 2. Run the evaluation

**Option A: HuggingFace-hosted (no Docker needed)**

```bash
python evaluate.py \\
    --agent scripted \\
    --agent-name "MyBot-v1" \\
    --agent-type Scripted \\
    --opponent Normal \\
    --games 10 \\
    --server https://openra-rl-openra-rl.hf.space
```

**Option B: Local server (Docker)**

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
