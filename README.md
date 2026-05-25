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

Standardized benchmark and leaderboard for LLM/RL agents playing
Red Alert through the OpenRA-Rust engine. The bench measures
isolated capabilities (combat micro, economy, scouting, long-horizon
planning, …) via scripted scenarios with a strict no-cheat bar.

## What's here

- **Scenario suite** — 200+ packs under `openra_bench/scenarios/packs/`,
  grouped into 11 capability families.
- **Rust engine** — the embedded-data engine at `OpenRA-Rust/` (see
  `OpenRA-Rust/VENDOR_DATA.md` for the unit-data provenance).
- **LLM evaluation harness** — `openra_bench.run_eval` runs an LLM
  agent through any subset of packs and produces structured
  per-turn + per-episode artifacts under `result/`.
- **Human evaluation UI** — `site/index.html` + `site/game_api.py`,
  a dark static-site app for humans to play scenarios with a
  **review-before-save** modal (no auto-save of accidental or
  contaminated runs).
- **Leaderboard browser** — `app.py` Gradio tabs for viewing
  existing leaderboard / capability-leaderboard / scenario catalog
  / battle viewer. The legacy Play / Playlist / Submit tabs were
  removed; the static site is the canonical human-eval UI.

## Quick Start

### Browse the leaderboard

```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
# Tabs: Leaderboard, Capability Leaderboard, Scenarios, Battle Viewer, About
```

### Run an LLM scenario evaluation

`python -m openra_bench.run_eval` drives the Rust engine through the
scenario packs in `openra_bench/scenarios/packs/` against an LLM
agent. Supported providers: `openrouter`, `vllm`, `openai`,
`together`, `bedrock`.

```bash
# OpenRouter / OpenAI / vLLM / Together (set the matching API_KEY first):
python -m openra_bench.run_eval \
    --packs openra_bench/scenarios/packs/perception-target-vs-fog.yaml \
    --levels easy --seeds 1 \
    --provider openrouter --model anthropic/claude-3.5-sonnet \
    --out eval_stats.json
```

For a full sweep:

```bash
python -m openra_bench.run_eval \
    --provider together --model "Qwen/Qwen3.5-9B" \
    --family all \
    --levels easy,medium,hard --seeds 1 \
    --concurrency 20 --qps 10 \
    --out data/runs/<run_name>/ \
    --playback data/runs/<run_name>/playback
```

After the sweep, consolidate into the canonical `result/` layout:

```bash
python3 tools/consolidate_results.py --input data/runs/ --output result/ --symlink
```

A 5-pack end-to-end smoke test of the Bedrock path lives in
[`docs/BEDROCK_SMOKE.md`](docs/BEDROCK_SMOKE.md).

### Run a human evaluation

The canonical human-play UI is the static site under `site/`. It
serves a dark-themed game-like interface with the **review-before-save**
modal: every game-over surfaces a turn-by-turn summary; the human
clicks **Save to dataset** (promotes the draft to `data/runs/...`)
or **Discard** (drops it). Until clicked, nothing lands on disk.

```bash
# 1. Start the FastAPI backend:
python -m site.game_api
# Listens on http://localhost:8000 by default

# 2. Serve the static UI:
cd site && python3 -m http.server 8765
# Open http://localhost:8765/index.html
```

Workflow: pick a scenario from the assigned annotator queue (or any
pack), play turn-by-turn, see end-of-game review modal, decide
save/discard, advance to the next assigned scenario.

The static site also has a no-server browse-only mode:

```bash
cd site && python3 -m http.server 8765
# Open http://localhost:8765/index.html
# (Browse + annotate without running games — no backend needed.)
```

Generate or refresh the static catalog after scenario changes:

```bash
python site/generate.py            # regenerate scenarios.json + thumbnails
python site/generate.py --dry-run  # print counts without writing
```

Deploy by copying `site/index.html` and `site/public/` to any static host.

See `docs/IMPLEMENTATION_NOTES.md` for full details on the static site.

### Validate the no-cheat bar

After editing any pack, run the bar validator to confirm no stall
(observe-only) policy WINS on any (pack, level, seed):

```bash
python3 tools/validate_pack_bar.py
python3 tools/analyze_pack_bar.py audits/pack_bar_status.csv
```

CI runs this on every PR (`.github/workflows/test.yml`) and fails on
any stall-wins cell.

### Running tests

```bash
# Engine wheel must be installed (see OpenRA-Rust/VENDOR_DATA.md):
cd OpenRA-Rust && maturin develop --release

# Bench suite:
python3 -m pytest tests/ -n auto --timeout=120 --tb=short
```

## Reproducibility

The bench's runtime behaviour is fully defined by ONE pair of
commits — the bench commit and the engine commit. See `VERSION` at
the repo root for the pinned pair. The engine embeds the RA unit
data (originally sourced from upstream OpenRA at SHA `0938a27`)
directly in its source tree — no runtime dependency on upstream.

Bumping either commit invalidates published bench numbers; re-run
the canonical sweep against the new pair.

## Scoring

| Component | Weight | Description |
|-----------|--------|-------------|
| Outcome (win / draw / loss) | 50% | The bar |
| Composite signals (kills, EV, exploration, …) | 30% | Per-pack dimension weights |
| Perception / Reasoning / Action diagnostics | 20% | Per-cell breakdown identifying weakest link |

See `openra_bench/scoring.py` for the full breakdown.

## Links

- [OpenRA-Rust engine repo](https://github.com/yxc20089/OpenRA-Rust)
- [Leaderboard Space](https://huggingface.co/spaces/openra-rl/OpenRA-Bench)
