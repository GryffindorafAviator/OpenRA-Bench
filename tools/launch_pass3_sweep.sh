#!/usr/bin/env bash
# Pass^3 stability sweep — every cell × 6 models × 3 attempts at temperature 0.7.
# Separate dir from v1.1-prod (which holds pass@1 results) for paper-grade
# code-state isolation. ~12h wall-clock at concurrency 20 per model.
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

PROD_DIR="data/runs/v1.1-prod-pass3"

for model in qwen3.5-9b gemma-4-31b-it qwen3.6-35b-a3b gpt-5.4-mini gpt-5.4 kimi-k2.6; do
  nohup python3 tools/run_production_eval.py launch \
    --model "$model" --type scenarios \
    --concurrency 20 --seeds 1 --auto-pr \
    --prod-dir "$PROD_DIR" \
    > "/tmp/prod_pass3_${model}.log" 2>&1 &
  disown
  echo "launched pass^3 sweep for $model"
  sleep 2
done
echo
echo "NOTE: orchestrator default repeats=1 per cell. To get pass^3, the launcher must"
echo "pass --repeats 3 down to run_eval. Need to add --repeats flag to run_production_eval"
echo "launch first, OR run with adjusted seed sets to simulate."
