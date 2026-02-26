"""OpenRA-Bench: Agent Leaderboard for OpenRA-RL.

A Gradio app that displays agent rankings, supports filtering by type
and opponent difficulty, and lets users run evaluations in-browser.

Run locally:
    python app.py

Deploy on HuggingFace Spaces:
    Push app.py, requirements.txt, data/, and README.md to your HF Space.
"""

import csv
import html
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from evaluate_runner import DEFAULT_SERVER, compute_composite_score, compute_game_metrics

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
    "Replay",
]


def _safe_agent_link(name: str, url) -> str:
    """Render agent name, optionally as a hyperlink. HTML-escaped to prevent XSS."""
    safe_name = html.escape(str(name))
    if pd.notna(url) and str(url).strip():
        url_str = str(url).strip()
        # Only allow http/https URLs — block javascript:, data:, etc.
        if url_str.startswith(("http://", "https://")):
            safe_url = html.escape(url_str, quote=True)
            return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_name}</a>'
    return safe_name


def _safe_replay_link(url) -> str:
    """Render replay download link. Filename is sanitized to prevent XSS."""
    if pd.notna(url) and str(url).strip():
        # Sanitize: only allow alphanumeric, dash, underscore, dot
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "", str(url).strip())
        if safe_name:
            escaped = html.escape(safe_name, quote=True)
            return f'<a href="/replays/{escaped}" download title="Download replay">&#11015;</a>'
    return ""


def load_data() -> pd.DataFrame:
    """Load leaderboard data from CSV."""
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Build agent name with optional hyperlink (XSS-safe)
    if "agent_url" in df.columns:
        df["Agent"] = df.apply(
            lambda r: _safe_agent_link(r.get("agent_name", ""), r.get("agent_url", "")),
            axis=1,
        )
    else:
        df["Agent"] = df["agent_name"].apply(lambda n: html.escape(str(n)))

    # Build replay download link (XSS-safe)
    if "replay_url" in df.columns:
        df["Replay"] = df["replay_url"].apply(_safe_replay_link)
    else:
        df["Replay"] = ""

    # Rename for display
    df = df.rename(columns={
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

    # Search by agent name (regex with fallback to literal on invalid patterns)
    if search and search.strip():
        patterns = [p.strip() for p in search.split(",") if p.strip()]
        mask = pd.Series([False] * len(df), index=df.index)
        for pattern in patterns:
            try:
                mask |= df["Agent"].str.contains(pattern, case=False, regex=True, na=False)
            except re.error:
                mask |= df["Agent"].str.contains(
                    re.escape(pattern), case=False, regex=True, na=False
                )
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


def _sanitize_csv_value(val):
    """Strip leading characters that trigger formula execution in spreadsheets."""
    if isinstance(val, str):
        while val and val[0] in ("=", "+", "-", "@", "\t", "\r", "\n"):
            val = val[1:]
        val = val.replace("\n", " ").replace("\r", " ")
    return val


# ── Rate Limiting ────────────────────────────────────────────────────────────

_submit_times: dict[str, list[float]] = defaultdict(list)
MAX_SUBMITS_PER_HOUR = 20


def _check_rate_limit(identifier: str = "global") -> tuple[bool, str]:
    """Simple in-memory rate limiter. Returns (allowed, error_message)."""
    now = time.time()
    times = _submit_times[identifier]
    _submit_times[identifier] = [t for t in times if now - t < 3600]
    if len(_submit_times[identifier]) >= MAX_SUBMITS_PER_HOUR:
        return False, "Rate limit exceeded (max 20 submissions per hour). Try again later."
    _submit_times[identifier].append(now)
    return True, ""


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
        "avg_game_length", "timestamp", "replay_url", "agent_url",
    ]
    safe_results = {k: _sanitize_csv_value(v) for k, v in results.items()}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(safe_results)


# ── Submission Handling ───────────────────────────────────────────────────────

MAX_REPLAY_SIZE = 10 * 1024 * 1024  # 10 MB

VALID_OPPONENTS = {"Beginner", "Easy", "Medium", "Normal", "Hard"}
VALID_AGENT_TYPES = {"Scripted", "LLM", "RL"}
REQUIRED_FIELDS = [
    "agent_name", "agent_type", "opponent", "result",
    "ticks", "kills_cost", "deaths_cost", "assets_value",
]


def validate_submission(data: dict) -> tuple[bool, str]:
    """Validate an uploaded JSON submission.

    Returns (is_valid, error_message).
    """
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"

    if data["agent_type"] not in VALID_AGENT_TYPES:
        return False, (
            f"Invalid agent_type: {data['agent_type']}. "
            f"Must be one of: {', '.join(sorted(VALID_AGENT_TYPES))}"
        )

    if data["opponent"] not in VALID_OPPONENTS:
        return False, (
            f"Invalid opponent: {data['opponent']}. "
            f"Must be one of: {', '.join(sorted(VALID_OPPONENTS))}"
        )

    # Type checks for numeric fields
    for field in ("ticks", "kills_cost", "deaths_cost", "assets_value"):
        if not isinstance(data[field], (int, float)):
            return False, f"Field '{field}' must be a number"

    # String length limits
    if len(str(data["agent_name"])) > 100:
        return False, "agent_name must be 100 characters or fewer"

    # agent_url: optional, but must be http(s) if provided
    agent_url = str(data.get("agent_url", "")).strip()
    if agent_url and not agent_url.startswith(("http://", "https://")):
        return False, "agent_url must be an HTTP(S) URL"
    if len(agent_url) > 500:
        return False, "agent_url must be 500 characters or fewer"

    return True, ""


def _score_from_submission(data: dict) -> dict:
    """Build a CSV-ready results dict from a validated submission."""
    game_result = {
        "result": data.get("result", ""),
        "win": data.get("win", data.get("result") == "win"),
        "ticks": data.get("ticks", 0),
        "kills_cost": data.get("kills_cost", 0),
        "deaths_cost": data.get("deaths_cost", 0),
        "kd_ratio": data.get("kd_ratio", 0),
        "assets_value": data.get("assets_value", 0),
        "cash": data.get("cash", 0),
    }
    score = compute_composite_score([game_result])
    kills = data.get("kills_cost", 0)
    deaths = data.get("deaths_cost", 0)
    games = data.get("games", 1)

    return {
        "agent_name": data["agent_name"],
        "agent_type": data["agent_type"],
        "opponent": data["opponent"],
        "games": games,
        "win_rate": round(100.0 * (1 if data.get("win") else 0) / max(games, 1), 1),
        "score": round(score, 1),
        "avg_kills": kills,
        "avg_deaths": deaths,
        "kd_ratio": round(kills / max(deaths, 1), 2),
        "avg_economy": data.get("assets_value", 0),
        "avg_game_length": data.get("ticks", 0),
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d"))[:10],
        "replay_url": "",
        "agent_url": data.get("agent_url", ""),
    }


def handle_upload(json_file, replay_file) -> tuple[str, pd.DataFrame]:
    """Process an uploaded bench submission JSON + optional replay."""
    if json_file is None:
        return "Please upload a JSON file.", add_type_badges(load_data())

    allowed, err = _check_rate_limit()
    if not allowed:
        return err, add_type_badges(load_data())

    try:
        with open(json_file.name) as f:
            data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return f"Invalid JSON: {e}", add_type_badges(load_data())

    is_valid, error = validate_submission(data)
    if not is_valid:
        return f"Validation error: {error}", add_type_badges(load_data())

    results_row = _score_from_submission(data)

    # Save replay if provided
    if replay_file is not None:
        import shutil

        orig = Path(replay_file.name)
        if orig.stat().st_size > MAX_REPLAY_SIZE:
            return "Replay file too large (max 10 MB).", add_type_badges(load_data())
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "", data["agent_name"].replace("/", "_").replace(" ", "_"))[:30]
        replay_name = f"replay-{slug}-{ts}.orarep"
        shutil.copy2(str(orig), SUBMISSIONS_DIR / replay_name)
        results_row["replay_url"] = replay_name

    save_submission(results_row)

    return (
        f"Submitted! **{data['agent_name']}** ({data['agent_type']}) "
        f"vs {data['opponent']}: score **{results_row['score']}**",
        add_type_badges(load_data()),
    )


def handle_api_submit(json_data: str) -> str:
    """API endpoint: accept JSON string submission. Used by CLI auto-upload."""
    allowed, err = _check_rate_limit()
    if not allowed:
        return err

    try:
        data = json.loads(json_data)
    except (json.JSONDecodeError, Exception) as e:
        return f"Invalid JSON: {e}"

    is_valid, error = validate_submission(data)
    if not is_valid:
        return f"Validation error: {error}"

    results_row = _score_from_submission(data)
    save_submission(results_row)

    return (
        f"OK: {data['agent_name']} ({data['agent_type']}) "
        f"vs {data['opponent']}: score {results_row['score']}"
    )


def handle_api_submit_with_replay(json_data: str, replay_file) -> str:
    """API endpoint: accept JSON + replay file. Used by CLI with --replay."""
    allowed, err = _check_rate_limit()
    if not allowed:
        return err

    try:
        data = json.loads(json_data)
    except (json.JSONDecodeError, Exception) as e:
        return f"Invalid JSON: {e}"

    is_valid, error = validate_submission(data)
    if not is_valid:
        return f"Validation error: {error}"

    results_row = _score_from_submission(data)

    # Save replay if provided
    if replay_file is not None:
        import shutil

        orig = Path(replay_file) if isinstance(replay_file, str) else Path(replay_file.name)
        if orig.exists() and orig.stat().st_size > MAX_REPLAY_SIZE:
            return "Replay file too large (max 10 MB)"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^a-zA-Z0-9_-]", "", data["agent_name"].replace("/", "_").replace(" ", "_"))[:30]
        replay_name = f"replay-{slug}-{ts}.orarep"
        shutil.copy2(str(orig), SUBMISSIONS_DIR / replay_name)
        results_row["replay_url"] = replay_name

    save_submission(results_row)

    return (
        f"OK: {data['agent_name']} ({data['agent_type']}) "
        f"vs {data['opponent']}: score {results_row['score']}"
    )


# ── UI ────────────────────────────────────────────────────────────────────────

ABOUT_MD = """
## What is OpenRA-Bench?

**OpenRA-Bench** is a standardized benchmark for evaluating AI agents playing
[Red Alert](https://www.openra.net/) through the
[OpenRA-RL](https://openra-rl.dev) environment.

### Evaluation Protocol

- **Game**: Red Alert (OpenRA engine)
- **Format**: 1v1 agent vs built-in AI
- **Opponents**: Beginner, Easy, Medium, Normal, Hard difficulty
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
---

## Other Submission Methods

### CLI Auto-Upload

Set `BENCH_URL` in your OpenRA-RL config and results upload automatically
after each game:

```yaml
# config.yaml
agent:
  bench_url: "https://openra-rl-openra-bench.hf.space"
```

### CLI Manual Upload

Upload a previously exported bench JSON:

```bash
python -m openra_env.bench_submit ~/.openra-rl/bench-exports/bench-*.json
```

### Batch Evaluation (10+ games)

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
    --server http://localhost:8000
```

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `--agent` | Agent type: `scripted`, `llm`, `mcp`, `custom` |
| `--agent-name` | Display name on the leaderboard |
| `--agent-type` | Category: `Scripted`, `LLM`, `RL` |
| `--opponent` | AI difficulty: `Beginner`, `Easy`, `Medium`, `Normal`, `Hard` |
| `--games` | Number of games (minimum 10) |
| `--server` | OpenRA-RL server URL (local or HuggingFace-hosted) |

### Custom Agents

Implement the standard `reset/step` loop:

```python
from openra_env.client import OpenRAEnv
from openra_env.models import OpenRAAction

async with OpenRAEnv("http://localhost:8000") as env:
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
                        choices=["All", "Beginner", "Easy", "Medium", "Normal", "Hard"],
                        value="All",
                        label="Opponent",
                        scale=1,
                    )

                leaderboard = gr.Dataframe(
                    value=initial_df,
                    datatype=[
                        "number",    # Rank
                        "html",      # Agent (may contain hyperlink)
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
                        "html",      # Replay (download link)
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
                gr.Markdown(
                    "## Upload Results\n\n"
                    "Upload a bench export JSON from your OpenRA-RL game. "
                    "After each game, the agent saves a JSON file to "
                    "`~/.openra-rl/bench-exports/`."
                )
                with gr.Row():
                    json_upload = gr.File(
                        label="Bench export JSON",
                        file_types=[".json"],
                        scale=3,
                    )
                    replay_upload = gr.File(
                        label="Replay file (optional)",
                        file_types=[".orarep"],
                        scale=2,
                    )
                submit_btn = gr.Button("Submit Results", variant="primary")
                submit_output = gr.Markdown()

                submit_btn.click(
                    fn=handle_upload,
                    inputs=[json_upload, replay_upload],
                    outputs=[submit_output, leaderboard],
                )

                # API endpoint for CLI auto-upload (JSON only)
                api_json_input = gr.Textbox(visible=False)
                api_result = gr.Textbox(visible=False)
                api_btn = gr.Button(visible=False)
                api_btn.click(
                    fn=handle_api_submit,
                    inputs=[api_json_input],
                    outputs=[api_result],
                    api_name="submit",
                )

                # API endpoint for CLI upload with replay
                api_json_input2 = gr.Textbox(visible=False)
                api_replay_input = gr.File(visible=False)
                api_result2 = gr.Textbox(visible=False)
                api_btn2 = gr.Button(visible=False)
                api_btn2.click(
                    fn=handle_api_submit_with_replay,
                    inputs=[api_json_input2, api_replay_input],
                    outputs=[api_result2],
                    api_name="submit_with_replay",
                )

                gr.Markdown(SUBMIT_MD)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(allowed_paths=[str(SUBMISSIONS_DIR)])
