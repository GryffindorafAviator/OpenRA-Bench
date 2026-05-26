#!/usr/bin/env bash
# Full 1v1 grid: 15 unordered pairs (incl 5 self) × 8 episodes (4 seeds × side-swap).
# 6 models (glm-4.6v dropped, added). C(6,2) + 6 self = 15 + 6 = 21 pairs.
# Each pair is launched as a separate `run_production_eval launch --type 1v1` so
# the orchestrator's auto-PR fires on completion.
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a

MODELS=(qwen3.5-9b gemma-4-31b-it qwen3.6-35b-a3b gpt-5.4-mini gpt-5.4)

# Per-model opponent spec for cross-pair launches
spec_for() {
  case "$1" in
    qwen3.5-9b)        echo "together:Qwen/Qwen3.5-9B" ;;
    gemma-4-31b-it)    echo "openrouter:google/gemma-4-31b-it" ;;
    qwen3.6-35b-a3b)   echo "together:together_sso/Qwen/Qwen3.6-35B-A3B-FP8-46d45bad" ;;
    gpt-5.4-mini)      echo "openai:gpt-5.4-mini-2026-03-17" ;;
    gpt-5.4)           echo "openai:gpt-5.4-2026-03-05" ;;
    *) echo "scripted:stall" ;;
  esac
}

# Iterate every unordered pair (i ≤ j) — 15 pairs including self
launched=0
for ((i=0; i<${#MODELS[@]}; i++)); do
  for ((j=i; j<${#MODELS[@]}; j++)); do
    a="${MODELS[$i]}"
    b="${MODELS[$j]}"
    opp=$(spec_for "$b")
    log="/tmp/prod_1v1_${a}_vs_${b}.log"
    echo "launching: $a vs $b (opp=$opp)"
    nohup python3 tools/run_production_eval.py launch \
      --model "$a" --type 1v1 \
      --opponent "$opp" \
      --concurrency 20 --seeds 1,2,3,4 --auto-pr \
      --prod-dir "data/runs/v1.1-prod-1v1/${a}_vs_${b}" \
      > "$log" 2>&1 &
    disown
    launched=$((launched + 1))
    sleep 2  # avoid manifest race
  done
done

echo
echo "launched $launched pairs (= C(5,2)+5 = 15)"
sleep 5
pgrep -af "openra_bench.run_eval" | wc -l | sed 's/^/  alive run_eval children: /'
