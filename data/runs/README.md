# `data/runs/` — Phase 4 paper data

This directory holds the per-run JSONL + per-turn PNG output of
`scripts/collect_eval_data.py`. JSONLs can be hundreds of MB per
run (full untruncated obs + system prompt + model request +
response per turn, plus minimap base64); PNGs are ~150 KB each.

**Heavy payloads (`*.jsonl`, `*.png`) are intentionally NOT
committed** to keep the repo lean. The summary metadata
(`_summary.json`, `_invocation.json`) IS committed — it records
what cells were run, how long they took, and where the JSONL
files live locally.

To re-create the data: rerun the same `scripts/collect_eval_data.py`
invocation; `--resume` will skip any cells whose JSONL still has
a `terminal:` line locally.

## Runs in this session

- `_smoke_16cell/` — 16-cell plumbing verification (Qwen3.5-9B)
- `paper-v1-engine-feature-packs/` — main collection (12 packs × 2
  levels × 2 seeds × 5 models; in flight, ~69 cells captured of
  240 planned at session checkpoint)
- `paper-v1-plus-medium/` — Qwen3.6-Plus F1-scale-test (8/8)
- `paper-v1-gemma-medium/` — gemma-4-31B-it F1-cross-model
  control (5/6 in flight)
- `_archive_flash-gemma-orphan/` — orphaned cells from an early
  flash run that errored; kept for audit reference
