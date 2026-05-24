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
- **Mission Player**: Static game-like website for browsing, annotating, and reviewing scenarios
- **Bilingual**: English and Chinese scenario instructions generated deterministically

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

### Run an LLM scenario eval

`python -m openra_bench.run_eval` drives the Rust engine through the
scenario packs in `openra_bench/scenarios/packs/` against an LLM
agent. Supported providers: `openrouter`, `vllm`, `openai`,
`together`, `bedrock`.

```bash
# OpenRouter / OpenAI / vLLM (set the matching API_KEY env var first):
python -m openra_bench.run_eval \
    --packs openra_bench/scenarios/packs/perception-target-vs-fog.yaml \
    --levels easy --seeds 1 \
    --provider openrouter --model anthropic/claude-3.5-sonnet \
    --out eval_stats.json

# AWS Bedrock — Claude Sonnet 4.6 via the cross-region inference profile.
# Auth is the standard boto3 credential chain (env / shared config /
# role); never hardcoded. The on-demand model id throws
# ValidationException; only the profile id below is callable.
aws sts get-caller-identity   # confirm credentials
python -m openra_bench.run_eval \
    --packs openra_bench/scenarios/packs/perception-target-vs-fog.yaml \
    --levels easy --seeds 1 \
    --provider bedrock \
    --model us.anthropic.claude-sonnet-4-6 \
    --bedrock-region us-west-2 \
    --out eval_stats.json
```

A 5-pack end-to-end smoke test of the Bedrock path lives in
[`docs/BEDROCK_SMOKE.md`](docs/BEDROCK_SMOKE.md).

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

## Mission Player (Static Site)

A game-like mission selection and annotation website in `site/`. No framework, no build step -- a single HTML file deployable to GitHub Pages.

### For players / annotators

Open `site/index.html` via any HTTP server:

```bash
cd site && python3 -m http.server 8765
# Open http://localhost:8765/index.html
```

Workflow: browse scenario cards, pick a mission, read bilingual objectives (EN/ZH toggle), switch difficulty (easy/medium/hard), annotate the map with point/region tools, tag and add notes, mark complete, navigate to next mission, export annotations as JSON.

### For maintainers

Generate or refresh static data after scenario changes:

```bash
python site/generate.py            # generate scenarios.json + map thumbnails
python site/generate.py --dry-run  # print counts without writing
```

Map thumbnails require the Rust engine wheel (`openra_train`). Without it, the site works with a placeholder map area; annotations still work on the placeholder.

Deploy by copying `site/index.html` and `site/public/` to any static host.

See `docs/IMPLEMENTATION_NOTES.md` for full details.

### Running tests

```bash
# Data pipeline + coverage invariant tests (Python)
python -m pytest tests/test_site.py tests/test_app.py -v

# E2E DOM interaction tests (Node.js + jsdom)
npm install   # first time only
node tests/test_site_e2e.mjs
```

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
