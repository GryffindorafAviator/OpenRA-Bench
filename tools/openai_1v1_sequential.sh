#!/usr/bin/env bash
# Sequential launcher for the 9 OpenAI-involved 1v1 pairs. Runs one pair to
# completion, then the next — so if the OpenAI quota burns out we have the
# completed pairs' data saved (rather than 9 half-finished sweeps that all
# stall together). Cheapest pairs first so we maximize completed-pair count
# before any potential budget hit.
#
# `pipefail` is LOAD-BEARING: the launcher invocation pipes to `tee` so the
# operator can watch progress. Without pipefail, `$?` reads tee's exit code
# (always 0) — masking a killed launcher. v1 footgun: 2 OpenAI pairs were
# silently skipped as "DONE" because the all-stop kill at 13:09 killed the
# launcher but the queue saw exit=0 and advanced.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

# Order: gpt-5.4-mini-involving FIRST (cheaper @ $0.75/$4.50 per 1M)
#        gpt-5.4-involving LAST (4× pricier)
PAIRS=(
  # 4× gpt-5.4-mini pairs (cheapest)
  "qwen3.5-9b gpt-5.4-mini together:Qwen/Qwen3.5-9B openai:gpt-5.4-mini-2026-03-17"
  "qwen3.6-35b-a3b gpt-5.4-mini together:together_sso/Qwen/Qwen3.6-35B-A3B-FP8-46d45bad openai:gpt-5.4-mini-2026-03-17"
  "gemma-4-31b-it gpt-5.4-mini together:together_sso/google/gemma-4-31B-it-f5dbf8ad openai:gpt-5.4-mini-2026-03-17"
  "gpt-5.4-mini gpt-5.4-mini openai:gpt-5.4-mini-2026-03-17 openai:gpt-5.4-mini-2026-03-17"
  # The cross-tier pair (gpt-5.4-mini vs gpt-5.4)
  "gpt-5.4-mini gpt-5.4 openai:gpt-5.4-mini-2026-03-17 openai:gpt-5.4-2026-03-05"
  # 4× gpt-5.4 pairs (most expensive — run last)
  "qwen3.5-9b gpt-5.4 together:Qwen/Qwen3.5-9B openai:gpt-5.4-2026-03-05"
  "qwen3.6-35b-a3b gpt-5.4 together:together_sso/Qwen/Qwen3.6-35B-A3B-FP8-46d45bad openai:gpt-5.4-2026-03-05"
  "gemma-4-31b-it gpt-5.4 together:together_sso/google/gemma-4-31B-it-f5dbf8ad openai:gpt-5.4-2026-03-05"
  "gpt-5.4 gpt-5.4 openai:gpt-5.4-2026-03-05 openai:gpt-5.4-2026-03-05"
)

mkdir -p /tmp/openai_1v1_seq
QUEUE_LOG=/tmp/openai_1v1_seq/queue.log
echo "=== sequential OpenAI 1v1 queue started $(date) ===" | tee -a "$QUEUE_LOG"

for entry in "${PAIRS[@]}"; do
  a=$(echo "$entry" | awk '{print $1}')
  b=$(echo "$entry" | awk '{print $2}')
  agent_spec=$(echo "$entry" | awk '{print $3}')
  opp_spec=$(echo "$entry" | awk '{print $4}')
  pair_id="${a}_vs_${b}"
  log="/tmp/openai_1v1_seq/${pair_id}.log"
  prod_dir="data/runs/v1.1-prod-1v1/${pair_id}"

  # Skip if already complete (8 episodes journaled)
  done_count=$(find "$prod_dir" -name "journal*.jsonl" 2>/dev/null | xargs grep -hcv '"_meta"' 2>/dev/null | head -1)
  done_count=${done_count:-0}
  if [ "$done_count" -ge 8 ]; then
    echo "[$(date +%H:%M:%S)] SKIP $pair_id (already $done_count/8 episodes)" | tee -a "$QUEUE_LOG"
    continue
  fi

  echo "[$(date +%H:%M:%S)] STARTING $pair_id (resume from $done_count/8)" | tee -a "$QUEUE_LOG"
  python3 tools/run_production_eval.py launch \
    --model "$a" --type 1v1 --opponent "$opp_spec" \
    --concurrency 10 --seeds 1,2,3,4 --auto-pr \
    --prod-dir "$prod_dir" \
    2>&1 | tee -a "$log"
  # ${PIPESTATUS[0]} reads the launcher's exit, not tee's. Re-verify after
  # by reading the journal — a "DONE exit=0" with 0 journal rows means the
  # launcher was killed externally; surface that to the operator clearly.
  ec=${PIPESTATUS[0]}
  final_count=$(find "$prod_dir" -name "journal*.jsonl" 2>/dev/null | xargs grep -hcv '"_meta"' 2>/dev/null | head -1)
  final_count=${final_count:-0}
  if [ "$ec" != "0" ] || [ "$final_count" -lt 8 ]; then
    echo "[$(date +%H:%M:%S)] INCOMPLETE $pair_id (exit=$ec  journal=$final_count/8)" | tee -a "$QUEUE_LOG"
  else
    echo "[$(date +%H:%M:%S)] DONE $pair_id (exit=$ec  journal=$final_count/8)" | tee -a "$QUEUE_LOG"
  fi
done

echo "=== sequential OpenAI 1v1 queue finished $(date) ===" | tee -a "$QUEUE_LOG"
