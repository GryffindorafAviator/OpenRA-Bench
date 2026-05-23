# `collect_eval_data.py` — Phase 4 paper-collection driver

Orchestrates per-cell audit-format eval runs across the Together AI
model roster, producing one JSONL (+ minimap PNG dir) per
`(model, pack, level, seed, fog_mode)` cell. Designed so a multi-hour,
multi-model collection can crash-resume losslessly and so every byte
needed for downstream paper analysis is captured at source — full
observations (including the `_raw` engine dict and the spatial
tensor), the literal HTTP request/response, engine warnings, per-turn
signals, and a `terminal:` block with outcome / wall-clock / token
totals.

## Data layout

```
<output-dir>/
  _invocation.json                    # the CLI args + cost estimate
  _summary.json                       # written on completion
  .logs/                              # per-cell stdout+stderr
    <model_safe>__<pack>__<level>__seed<N>__<fog>.log
    <model_safe>__<pack>__<level>__seed<N>__<fog>.stats.json
  <YYYYMMDD-HHMMSS>__<model_safe>/    # one dir per (timestamp, model)
    <pack>__<level>__seed<N>__<fog>.jsonl
    <pack>__<level>__seed<N>__<fog>/
      turn_001.png
      turn_002.png
      ...
```

The JSONL is one line per turn; each line has these fields (full
schema in `openra_bench/full_playback.py`):

| field             | description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `turn`            | int, 1-based turn index                                     |
| `tick`            | int, engine game tick at end of turn                        |
| `interrupt`       | str or null — engine signal name that fired this turn       |
| `obs`             | dict, full `RustObsAdapter.render_state()` (includes `_raw` and `spatial`) |
| `briefing`        | str, exactly the text the model received                    |
| `system_prompt`   | str, first turn only; subsequent turns = null               |
| `model_request`   | dict, literal `{url, body}` posted to the provider          |
| `model_response`  | dict, literal `{raw, text, tool_calls, reasoning, usage, finish_reason}` |
| `commands_issued` | list[str], `Command.repr()` per command parsed              |
| `engine_warnings` | list[str], from the Rust env's `info["warnings"]`           |
| `signals`         | dict, primitive signal snapshot (cash, kills, explored%, …) |
| `minimap_png`     | str or null, relative path to the per-turn PNG              |
| `done`            | bool, engine `done` flag                                    |
| `terminal`        | present ONLY on the final line; see below                   |

The `terminal` block has:

```json
{
  "outcome": "win|loss|draw",
  "final_obs": { ... },
  "wall_clock_seconds": 12.345,
  "total_tokens_in": 78901,
  "total_tokens_out": 1234,
  "manifest": { /* scoring/score metadata */ }
}
```

## Invocation

```bash
python3 scripts/collect_eval_data.py \
  --models Qwen/Qwen3.5-9B,Qwen/Qwen3.6-Plus,google/gemma-4-31B-it,moonshotai/Kimi-K2.6 \
  --packs all \
  --levels easy,medium,hard \
  --seeds 1,2,3,4 \
  --fog-modes vision \
  --run-label paper-collection-v1 \
  --output-dir data/runs/paper-collection-v1 \
  --parallel-cells 4 \
  --resume
```

* `--packs` accepts `all`, a comma list, `@file.txt` (one pack id per
  line), or a directory of `*.yaml`.
* `--fog-modes` accepts any subset of
  `structured,structured-clear,vision,vision-clear,image,image-clear`.
* `--parallel-cells` controls subprocess concurrency. Each cell is an
  isolated `python -m openra_bench.run_eval` invocation so a crash in
  one cell never aborts the rest of the run.

Required env var (set in your shell / `.env`): `TOGETHER_API_KEY`.

## Cost estimates

`--cost-estimate` prints per-model and total token / USD estimates
WITHOUT spawning any subprocesses. The estimator uses average turns
and tokens-per-turn from the May 2026 pilot runs
(`playback/pilot_perception/`, `playback/pilot_handoff/`):

* `avg_turns_per_cell = 18` (max_turns is typically 36; mean is ~half)
* `avg_prompt_tokens_per_turn = 4500` (briefing + image + codex,
  with the 16-turn sliding window)
* `avg_completion_tokens_per_turn = 250` (a tool call + brief
  reasoning)

Pricing snapshot (USD per 1M tokens, May 2026):

| model                  | in / M | out / M |
| ---------------------- | ------ | ------- |
| Qwen/Qwen3.5-9B        | $0.20  | $0.20   |
| Qwen/Qwen3.6-Plus      | $0.50  | $1.50   |
| qwen/qwen3.6-flash     | $0.18  | $0.18   |
| google/gemma-4-31B-it  | $0.25  | $0.25   |
| moonshotai/Kimi-K2.6   | $0.60  | $2.50   |

Update `_TOGETHER_PRICES` at the top of `scripts/collect_eval_data.py`
when Together's published rates change.

### Common run profiles (cost estimates)

| profile           | cells / model | total cells | est USD (4 models, 1 fog) |
| ----------------- | ------------- | ----------- | ------------------------- |
| **smoke**: 10 packs × 1 level × 1 seed × 1 fog        | 10  | 40   | ~$1     |
| **mini**:  50 packs × 3 levels × 1 seed × 1 fog       | 150 | 600  | ~$30    |
| **medium**: 50 packs × 3 levels × 4 seeds × 1 fog     | 600 | 2400 | ~$120   |
| **full**: 200 packs × 3 levels × 4 seeds × 1 fog      | 2400| 9600 | ~$480   |
| **perception sweep**: full × 6 fog modes              | 14400| 57600| ~$2900  |

Always run `--cost-estimate` before the real thing.

## Resume / re-run a failed cell

`--resume` scans the output dir and skips any cell whose JSONL is
complete (its last line has a `terminal:` field). Cells that crashed
mid-run leave a `<stem>.jsonl.partial` behind for forensics; the
final `<stem>.jsonl` is NOT created, so resume correctly retries them.

To force a re-run of one cell, delete its `<stem>.jsonl` (and the
sibling PNG dir if you want a fresh image series). The next
`--resume` invocation will re-spawn it.

For diagnostics, every cell's stdout+stderr is captured to
`<output-dir>/.logs/<cell-id>.log`. Tail one to see exactly what
`python -m openra_bench.run_eval` printed.

## Loading the data for paper analysis

```python
import json
from pathlib import Path

def iter_cells(run_dir):
    """Yield (path, lines) per cell — lines are list[dict]."""
    for p in sorted(Path(run_dir).glob("**/*.jsonl")):
        if p.name.startswith("_") or p.name.endswith(".partial"):
            continue
        with open(p) as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        yield p, lines

for path, lines in iter_cells("data/runs/paper-collection-v1"):
    term = lines[-1].get("terminal", {})
    outcome = term.get("outcome", "?")
    n_turns = len(lines)
    cost = (
        term.get("total_tokens_in", 0) / 1e6 * 0.5
        + term.get("total_tokens_out", 0) / 1e6 * 1.5
    )
    print(f"{path.stem:<60} {outcome:<5} turns={n_turns:>3} ~${cost:.3f}")
```

The viewer at `scripts/view_playback.py` transparently understands
the audit JSONL format alongside the legacy `seed<N>/` dirs — point
it at `data/runs/<run-label>` and it picks up both shapes.

## Caveats / known limits

* `--repeats > 1` currently shares the JSONL stem; for paper-grade
  collection keep it at 1 (one cell == one deterministic seed). The
  cell-level reliability metric (pass^k) belongs in a separate sweep
  with distinct `--seeds`.
* `--full-playback` runs ALONGSIDE the legacy `Playback` — pass both
  `--playback` and `--full-playback` to `run_eval` if you want the
  human-readable viewer files AND the audit JSONL. The collector
  script only emits the audit format (the playback dir is what the
  viewer reads natively).
* Engine warnings reflect the `info["warnings"]` list emitted by the
  Rust env at each step; the bench does NOT attach a model-level
  warning channel (the model is judged only by its commands).
* `model_response.raw` carries the entire provider response JSON,
  which on Together's side includes a `usage` block; downstream
  paper analysis should pull token counts from there rather than
  trusting any aggregate, because the per-turn breakdown is the
  authoritative source.
