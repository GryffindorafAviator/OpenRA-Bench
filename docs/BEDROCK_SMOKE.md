# AWS Bedrock smoke test — Claude Sonnet 4.6

End-to-end validation of the `BedrockProvider` (`openra_bench/providers.py`)
against a real AWS Bedrock endpoint. One run per capability bucket; this
is a wiring proof, not a leaderboard entry.

## Setup

* `boto3` available (`pip install boto3`).
* AWS credentials present in the boto3 credential chain
  (`aws sts get-caller-identity` succeeds). The bench never reads
  credentials directly — boto3 picks them up from env / shared config /
  IAM role.
* Engine wheel built (`cd OpenRA-Rust && PATH=$HOME/.cargo/bin:$PATH
  maturin develop --release`).
* Sonnet 4.6 is exposed via the `us.anthropic.claude-sonnet-4-6`
  cross-region inference profile served from `us-west-2`. The on-demand
  model id (`anthropic.claude-sonnet-4-6`) returns
  `ValidationException` — use the profile id.

Verify the model is callable directly:

```bash
aws bedrock-runtime converse \
    --region us-west-2 \
    --model-id us.anthropic.claude-sonnet-4-6 \
    --messages '[{"role":"user","content":[{"text":"hi"}]}]'
```

## Run

One pack per capability bucket, easy tier × seed 1:

```bash
for pack in perception-target-vs-fog combat-focus-fire-priority \
            def-walls-vs-towers build-production-throughput-multibuilding \
            mcv-deploy-near-resource; do
    python -m openra_bench.run_eval \
        --packs openra_bench/scenarios/packs/${pack}.yaml \
        --levels easy --seeds 1 \
        --provider bedrock \
        --model us.anthropic.claude-sonnet-4-6 \
        --bedrock-region us-west-2 \
        --out /tmp/bedrock_smoke_${pack}.json
done
```

## Results (1 episode each, easy tier, seed 1)

Captured 2026-05-23 from `/tmp/bedrock_smoke_*.json`.

| Pack | Bucket | Outcome | Turns | Composite | Prompt tok | Completion tok | Tool calls |
|------|--------|---------|------:|----------:|-----------:|---------------:|-----------:|
| `perception-target-vs-fog` | perception | **win** | 11 | 0.809 | 52,806 | 2,149 | 11 |
| `combat-focus-fire-priority` | combat | draw | 30 | 0.633 | 294,591 | 10,958 | 30 |
| `def-walls-vs-towers` | defense | loss | 37 | 0.238 | 276,402 | 7,272 | 37 |
| `build-production-throughput-multibuilding` | build/economy | **win** | 30 | 0.782 | 212,917 | 5,684 | 30 |
| `mcv-deploy-near-resource` | mcv | loss | 60 | 0.261 | 485,203 | 16,800 | 60 |
| **Total** | | 2W / 1D / 2L | 168 | — | 1,321,919 | 42,863 | 168 |

Estimated Bedrock spend at list price ($3 / MTok input, $15 / MTok output
for Sonnet): ~$4.61 across all five runs.

Every episode produced exactly one tool call per turn (the agent's
contract — see `openra_bench/agent.py`), the engine processed each
tool call without parser errors, and `score_episode` evaluated win /
loss / draw against the pack's win / fail predicates. No exceptions, no
schema mismatches. The wiring is correct.

## What this validates

1. Scenario YAML loads and compiles.
2. `BedrockProvider.complete()` round-trips:
   - System prompt → `system: [{text}]`.
   - Multimodal user briefing (text + PNG minimap data URL) → Bedrock
     `[{text}, {image: {format: png, source: {bytes}}}]`.
   - Assistant tool calls → Bedrock `toolUse` content blocks (with
     dict `input`, not the JSON-string OpenAI uses).
   - Tool replies → Bedrock user `toolResult` blocks (`toolUseId`
     preserved from the agent's `c0`/`c1` ids).
   - Adjacent same-role turns merge to satisfy Bedrock's strict
     user/assistant alternation.
   - Bedrock `output.message.content` parses back to `ChatReply`
     with `text`, `tool_calls=[{name, arguments}]`, and
     `usage={prompt_tokens, completion_tokens}`.
3. Engine accepts the resulting `Command` objects.
4. Win / fail predicates evaluate to a real outcome (no draw
   degeneracy on a real WIN / LOSS).
5. `eval_stats.json` lands with the expected schema (see
   `openra_bench/run_eval.py:_finalize`).

## Out of scope

* Per-capability score interpretation. Two losses and a draw out of
  five easy-tier runs is normal model variance — the goal here is the
  wire, not a leaderboard ranking. A full sweep (all packs × tiers ×
  seeds × repeats for CI) is a follow-up.
* OpenRouter / vLLM / together paths — those were not modified and
  remain on the existing `OpenAICompatibleProvider`.
* Non-`us-west-2` regions or other inference profiles. Adjust
  `--bedrock-region` and `--model` accordingly if the calling account
  has a different profile granted.
