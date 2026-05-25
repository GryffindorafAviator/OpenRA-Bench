# Production-eval setup on a fresh machine

Reproduce the v1.1 paper-baseline campaign on a new machine, including
adding a Bedrock model to the eval matrix at the same rigor (pass^3
scenarios + full 1v1 grid).

## 1. Clone + build the engine wheel

```bash
git clone https://github.com/yxc20089/OpenRA-Bench.git
git clone https://github.com/yxc20089/OpenRA-Rust.git
cd OpenRA-Rust
# Build the bench wheel (Rust + maturin)
PATH=$HOME/.cargo/bin:/opt/anaconda3/bin:$PATH maturin develop --release
# verify
python3 -c "import openra_train; print(openra_train.__file__)"
```

Engine main is the reference (currently HEAD `7b64e46` w/ PRs #18, #19, #20 landed).
Wheel rebuild takes ~3-5 min.

## 2. Bench checkout

```bash
cd OpenRA-Bench
git checkout v11-sweep-audit-fixes
pip install -r requirements.txt
# Or: pip install -e .  if there's a setup.py
```

## 3. API keys — `.env` (gitignored)

```bash
cat > .env <<EOF
TOGETHER_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
# Bedrock uses standard AWS creds:
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-west-2     # honored as default; per-model bedrock_region override available
EOF
```

`boto3` is a soft dep — install when adding a Bedrock model:
```bash
pip install boto3
```

## 4. Add a Bedrock model to the eval matrix

Edit `tools/run_production_eval.py`, find the `MODELS` tuple, add an entry:

```python
MODELS = (
    # ... existing 5 ...
    ("claude-3.5-sonnet",     "bedrock",    "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    # any other Bedrock model — Anthropic, Meta Llama, AI21, Cohere, Mistral all supported
)
```

The slug (`claude-3.5-sonnet`) is the filesystem-safe identity. The
model_id is the Bedrock model invocation string. Region is read from
`AWS_REGION` env (or `ProviderConfig.bedrock_region`, default `us-west-2`).

## 5. Run the v1.1 scenarios sweep for the new model

```bash
# Scenarios first — single seed × all levels × pass^3 (temperature 0.7)
python3 tools/run_production_eval.py launch \
    --model claude-3.5-sonnet \
    --type scenarios \
    --concurrency 20 \
    --seeds 1 \
    --repeats 3 \
    --auto-pr
```

`--auto-pr` opens a PR to `KaiserWhoLearns/RedAlertBenchPaper` with the
results when the sweep finishes. The summary uses the same percentage
formatting as the other models' PRs (#2-#7).

## 6. After scenarios, run the 1v1 grid

```bash
# Edit tools/launch_1v1_grid.sh to add the new model + its opponent spec
# Then:
bash tools/launch_1v1_grid.sh
```

## 7. Status / resume / monitor

```bash
# Status across the campaign
python3 tools/run_production_eval.py status

# Resume a crashed sweep (the journal is the source of truth)
python3 tools/run_production_eval.py launch --model <slug> --type scenarios --auto-pr
# (the orchestrator's `--resume` is default; `--ignore-run-id` is forwarded
# unconditionally so a relaunch picks up where the previous one died.)
```

## 8. Known-good run config

The 6 existing scenarios PRs in `KaiserWhoLearns/RedAlertBenchPaper`
record the exact config used. Excerpt (every PR carries this block):

```
## run configuration
- temperature: 0.7
- max_tokens: 1024 per completion
- max_retries: 5 per call (exponential backoff base=1.0s, cap=30.0s)
- timeout: 120.0s per call
- fog mode: vision (image-primary minimap + text briefing)
- max_history_turns: 16 (sliding wire-history window)
- concurrency: 20 parallel cells (adaptive halving on >10% error rate over last 20)
- seeds: 1 × levels easy,medium,hard
```

## Known gotchas

- **Together dedicated endpoints** (gemma, qwen3.6) auto-stop after inactivity.
  Restart via `client.endpoints.update(endpoint_id=..., state="STARTED")` from
  the `together` SDK; takes 3-5 min to warm up.
- **OpenAI gpt-5/o-series** require `max_completion_tokens` not `max_tokens`.
  Already handled in `providers.py` via a model-id prefix sniff.
- **OpenRouter glm-4.6v** had 23% JSONDecodeError rate under concurrency=20.
  Retry layer now catches malformed-JSON → transient retry. Dropped from
  v1.1 lineup for cost; can re-enable by uncommenting the MODELS entry.
- **Together moonshotai/Kimi-K2.6** hits 429 throttling even at
  concurrency=1. Excluded from v1.1.
- **Packs with `configs:` block** (`adversarial-duel`, `adversarial-1v1-macro`)
  pin fog_mode per config. If your old journal lacks `fog_mode` on those
  records, see the fog-key patch in commit history — runs after that
  commit auto-populate the field correctly.
