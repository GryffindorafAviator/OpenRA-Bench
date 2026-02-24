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
# Start OpenRA-RL server
cd /path/to/OpenRA-RL
docker compose up openra-rl

# Run benchmark
cd /path/to/OpenRA-Bench
python evaluate.py \
    --agent scripted \
    --agent-name "MyBot-v1" \
    --opponent Normal \
    --games 10
```

### Submit results

1. Fork this repo
2. Run evaluation (appends to `data/results.csv`)
3. Open a PR with your results

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
- [HuggingFace Space](https://huggingface.co/spaces/openra-rl/OpenRA-Bench)
