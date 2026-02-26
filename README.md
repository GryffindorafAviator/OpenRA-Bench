---
title: OpenRA-Bench
emoji: 🎮
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: true
license: gpl-3.0
---

# OpenRA-Bench

Standardized benchmark and leaderboard for AI agents playing Red Alert through [OpenRA-RL](https://openra-rl.dev).

## Features

- **Leaderboard**: Ranked agent comparison with composite scoring
- **Filtering**: By agent type (Scripted/LLM/RL) and opponent difficulty
- **Evaluation harness**: Automated N-game benchmarking with metrics collection
- **OpenEnv rubrics**: Composable scoring (win/loss, military efficiency, economy)
- **Replay verification**: Replay files linked to leaderboard entries

## Quick Start

### View the leaderboard

```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```

### Run an evaluation

```bash
# Against the HuggingFace-hosted environment (no Docker needed)
python evaluate.py \
    --agent scripted \
    --agent-name "MyBot-v1" \
    --opponent Normal \
    --games 10 \
    --server https://openra-rl-openra-rl.hf.space

# Or against a local Docker server
python evaluate.py \
    --agent scripted \
    --agent-name "MyBot-v1" \
    --opponent Normal \
    --games 10 \
    --server http://localhost:8000
```

### Submit results

**Via CLI (recommended):**

```bash
pip install openra-rl
openra-rl bench submit result.json
openra-rl bench submit result.json --replay game.orarep --agent-name "MyBot" --agent-url "https://github.com/user/mybot"
```

Results from `openra-rl play` are auto-submitted after each game.

**Via PR:**

1. Fork this repo
2. Run evaluation (appends to `data/results.csv`)
3. Open a PR with your results

### Agent identity

Customize your leaderboard entry:

| Field | Description |
|-------|-------------|
| `agent_name` | Display name (e.g. "DeathBot-9000") |
| `agent_type` | `Scripted`, `LLM`, or `RL` |
| `agent_url` | GitHub/project URL — renders as a clickable link on the leaderboard |

### Replay downloads

Entries submitted with a `.orarep` replay file show a download link in the Replay column. Replays are stored on the Space and served at `/replays/<filename>`.

### API endpoints

The Gradio app exposes these API endpoints (Gradio 5+ SSE protocol):

| Endpoint | Description |
|----------|-------------|
| `submit` | Submit JSON results (no replay) |
| `submit_with_replay` | Submit JSON + replay file |
| `filter_leaderboard` | Query/filter leaderboard data |

## Scoring

| Component | Weight | Description |
|-----------|--------|-------------|
| Win Rate | 50% | Games won / total games |
| Military Efficiency | 25% | Kill/death cost ratio (normalized) |
| Economy | 25% | Final asset value (normalized) |

## Links

- [OpenRA-RL Documentation](https://openra-rl.dev)
- [OpenRA-RL GitHub](https://github.com/yxc20089/OpenRA-RL)
- [OpenEnv Framework](https://huggingface.co/openenv)
- [Leaderboard Space](https://huggingface.co/spaces/openra-rl/OpenRA-Bench)
- [Environment Space](https://huggingface.co/spaces/openra-rl/OpenRA-RL)
