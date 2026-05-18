# OpenRA-Bench — Scenario Audit & Gap Analysis

Research + audit. No code changes. Prepared 2026-05-17.

---

## Executive Summary

OpenRA-Bench currently ships **85 scenario files** (17 hand-authored
scenario families + 68 procedurally generated `cat-*` packs across 12
categories, ~200 difficulty levels + 1 TEMPLATE). The design is
unusually well-grounded: every win condition is a verifiable predicate,
difficulty scales by *decision hardness* (info down, decoys up, clock
down, attrition cap) rather than raw numbers, and each pack carries a
`real_world_meaning` + `robotics_analogue`. The anti-memorization
discipline (procedural variation, held-out seeds, generalization-gap
metric) is consistent with current best practice (Procgen, SMACv2,
ARC-AGI). The "RTS reasoning transfers" hypothesis is *directly
supported by published evidence*: lmgame-Bench (arXiv:2505.15146)
showed RL on Sokoban/Tetris lifted Blocksworld (+≥10pts planning) and
WebShop (+6pts multi-turn decisions) but **not** math/coding — i.e.
transfer is real but axis-specific (spatial / planning / embodied),
exactly the axes this suite targets.

**The core finding of this audit:** the capability *taxonomy* is
sound and the transfer story is defensible, but the **coverage is
badly skewed**. Frontier models in 2026 are differentiated almost
entirely on (a) agentic **tool-use / function-calling fidelity** under
strict APIs (BFCL V4, τ²-bench — GPT-5.5 reports 98% τ²-Telecom,
Kimi-K2.6 50% Toolathlon), (b) **long-horizon multi-step agentic
execution** (Terminal-Bench 2.0 — GPT-5.5 82.7%, SWE-bench Verified
~89%), and (c) **fluid/abstract + spatial reasoning** (ARC-AGI-2 — the
single largest 2026 model delta; ERQA for embodied/spatial). OpenRA-Bench
is *strong* on spatial perception/exploration (FRONT/PERC — parallels
ERQA, the validated transfer target) and *adequate* on constrained
planning (PLAN/TECH/ECON — parallels PlanBench/Blocksworld). It is
**weak on adversarial/game-theoretic reasoning** (no opponent-modeling
scenario despite an RTS engine — the one capability an RTS bench is
uniquely positioned to own), **weak on long-horizon credit assignment**
(most clocks are short single-phase tasks, not 30+ step chains like
Terminal-Bench/OSWorld), and **structurally weak on the very thing
2026 leaderboards weight most: tool-use/instruction-following fidelity
under a strict action API** — this is *measured* (action-validity
sub-score) but no scenario *isolates* it as the primary objective.

Additionally, ~68 of 85 packs are the auto-generated `cat-*` family:
12 categories × ~3 levels × ~variants. Many `cat-c5/c6/c7/c8`
"budget-allocation / base-placement" packs are near-duplicates of each
other and of the hand-authored `economy-*` and `building-and-planning`
scenarios — high level count, low *distinct-capability* count. Several
economy packs degenerate to identical tests because harvest income is
0 on the only loadable map (documented engine-prereq S0/S1) — these
are honest but currently non-discriminating.

**Top recommendations (detail in §4):**
1. **Add an adversarial/opponent-modeling family** (the unique RTS
   value prop; parallels game-theoretic + StarCraft-II-Arena evals).
   Activate the Phase-1 1v1 path — this is the highest-leverage gap.
2. **Add a strict-action-API instruction-following family** isolating
   tool-call fidelity (parallels BFCL V4 / τ²-bench — the most
   leaderboard-weighted 2026 capability).
3. **Add 2–3 genuinely long-horizon scenarios** (multi-phase,
   40k+ ticks, credit assignment across opening→tech→assault) to
   parallel Terminal-Bench/OSWorld long-task tracks.
4. **Cut/merge the redundant `cat-*` over-generation** to ~3 distinct
   levels per *distinct* decision; the level *count* is inflating
   apparent coverage without adding capability breadth.
5. **Pre-register the transfer panel against ERQA (spatial/embodied),
   Blocksworld/PlanBench (planning), and BFCL/τ²-bench (tool-use)** —
   ERQA is the empirically-validated correlate for the headline claim;
   BFCL validates the under-tested action-fidelity axis.

---

## STEP 1 — Scenario Inventory

### 1.1 Counts

| Group | Files | Notes |
|---|---|---|
| Hand-authored scenario families | 17 | perception/reasoning/action/strategy/economy |
| Generated catalog packs (`cat-*`) | 68 | 12 categories (c1–c12), 5–6 variants × 3 levels each |
| TEMPLATE | 1 | scaffold, not a scenario |
| **Total scenario files** | **85** | ~200 difficulty levels |

The 12 generated categories (from `SCENARIO_CATALOG.md`, verified
against the `cat-*` files): C1 Frontier Scouting (FRONT), C2 Threat
Enumeration (PERC), C3 Tech Critical Path (TECH), C4 Power-Budget
Online (PLAN), C5 Budget Allocation (ECON), C6 Time-Boxed Capital
Deploy (ECON), C7 Defensive-Direction Commit (PERC), C8 Base-Placement
& Staging (PERC), C9 Commit-vs-Retreat (RISK), C10 Force Coordination
(COORD), C11 Tempo/Timing Window (TEMPO), C12 Error Recovery/Replan
(RISK).

### 1.2 Capability buckets and what each actually tests

The Perception→Reasoning→Action chain is the framing
(`EVAL_STACK_PLAN.md`); every scenario is scored on all three links
with a `weakest_link` diagnostic (`openra_bench/scoring.py`), plus a
continuous `reward_vector` goal tracker (`openra_bench/goal_tracker.py`)
and pass/fail via pure predicates (`openra_bench/scenarios/win_conditions.py`).

| Bucket | Sub-skill | Scenarios | What it actually tests (predicate grammar) |
|---|---|---|---|
| **Perception** | spatial state-estimation under fog | `perception-frontier-reading`, `perception-target-vs-fog`, C2, C7, C8 | `explored_pct_gte`, `enemies/buildings_discovered_gte` under a clock + attrition cap. Read the minimap, steer sensing. |
| **Perception/Front** | which-unexplored-region-to-commit | `reasoning-frontier-commit`, C1 | Frontier selection (pathfinding solved; *choosing* the region is the test). Strong ERQA/Active-Neural-SLAM analogue. |
| **Reasoning** | constrained sequential planning | C3 (tech path), C4 (power budget), `building-and-planning` | Precondition-ordered build to a deadline with `power_surplus_gte≥0`. Blocksworld/PlanBench analogue. |
| **Reasoning/Econ** | resource allocation / multi-objective | `economy-investment`, `economy-time-box`, `economy-force-buildup`, C5, C6, `economy-harvest-*` | Convert a fixed budget into units AND infrastructure under a clock; commit a wide-vs-deep allocation. Harvest variants currently non-discriminating (income=0, see §1.3). |
| **Reasoning/Risk** | risk call / replan under partial info | `reasoning-risk-route`, `strategy-dilemma`, C9 (commit-vs-retreat), C12 (error recovery) | Safe-long vs short-lethal route; engage-vs-hold with attrition cap; rebuild after a setback (`after_ticks` gate). |
| **Action** | multi-unit coordination / execution | `action-multiunit-coordination`, `strategy-twobody`, C10 | Drive dispersed squads to converge in a region with `units_lost_lte:0`. Watch-And-Help / SMAC analogue. |
| **Action** | sequenced execution to deadline | `action-sequenced-execution`, `rush-hour`, `strategy-gauntlet` | Ordered build→deploy→reposition without dropping a step; multi-squad sweep-and-clear. |
| **Tempo** | timing-window discipline | C11 | `after_ticks:t0` then `units_killed_gte:N` within `T1` — act in a window, not before. |

All run on the single Rust-loadable map `rush-hour-arena`.

### 1.3 Honest limitations already documented in-repo

- **Harvest economy is non-functional** on the only loadable map: the
  Python schema rejects `mine`/`gmine`, the `.oramap` files seed no
  ore, and `silo` storage is hardcoded inert
  (`SCENARIO_BRAINSTORM.md` §"Verified ground-truth constraints").
  Consequence: `economy-harvest-timebox` / `economy-harvest-investment`
  / harvest variants degenerate to spend-only tests identical to the
  non-harvest economy packs. They are honestly tagged but **currently
  non-discriminating** — they inflate the economy-bucket count without
  adding a distinct capability. Blocked on engine-prereqs S0 (ore
  source) + S1 (silo storage).
- **No enemy-destruction predicate** — offensive success is proxied by
  `units_killed_gte` + positional `reach_region`/`building_in_region`.
- **No adversarial slot today** — the 1v1/Elo path is Phase-1, not
  built (`EVAL_STACK_PLAN.md`). The "Elo leaderboard" is currently
  only meaningful for the fixed-scenario composite, not head-to-head.

---

## STEP 2 — What Frontier Benchmarks Measure Today (2026)

Verified via the four required sources + targeted search. URLs in §5.

### 2.1 Required model pages

- **Qwen3.6-35B-A3B** (HF page, released ~Apr 2026). Reports a
  coding/agentic-heavy battery: SWE-bench Verified 73.4, SWE-bench
  Pro 49.5, Terminal-Bench 2.0 51.5, **TAU3-Bench 67.2**, Tool
  Decathlon 26.9, MCPMark 37.0; reasoning: MMLU-Pro 85.2, GPQA 86.0,
  AIME26 92.7; **spatial-intelligence block: EmbSpatialBench 84.3,
  RefSpatialBench 64.3, RefCOCO 92.0**. Takeaway: even a mid-size
  model now reports a *dedicated embodied-spatial benchmark block* and
  a *dedicated tool/agent block* — these are the differentiating axes.
- **Kimi-K2.6** (HF page). Agentic-first: OSWorld-Verified 73.1,
  **Toolathlon 50.0, MCPMark 55.9**, BrowseComp 83.2, Terminal-Bench
  2.0 66.7, SWE-bench Verified 80.2; reasoning AIME 2026 96.4,
  GPQA-Diamond 90.5. Tool-use / computer-use / long-horizon browsing
  dominate the card.
- **GPT-5.5** (OpenAI announcement; page is 403 to direct fetch,
  corroborated via search). Released ~Apr 23 2026. Headline numbers:
  **SWE-bench Verified 88.7**, SWE-bench Pro 58.6, **Terminal-Bench
  2.0 82.7 (SOTA)**, **τ²-bench Telecom 98.0**, **ARC-AGI-2 85.0
  (+11.7 over prior, the launch's single largest delta)**, GDPval
  84.9. The two most-emphasized capabilities: agentic
  coding/tool-orchestration and ARC-AGI-2 fluid reasoning.
- **Artificial Analysis** leaderboard. The page no longer exposes
  components inline, but the current **Intelligence Index v4.0** =
  10 evals: **GDPval-AA, τ²-Bench Telecom, Terminal-Bench Hard,
  SciCode, AA-LCR (long-context reasoning), AA-Omniscience, IFBench
  (instruction-following), Humanity's Last Exam, GPQA Diamond,
  CritPt**. Note the heavy weighting toward *agentic tool-use*
  (τ²-bench), *long-horizon terminal agency* (Terminal-Bench Hard),
  and *instruction-following fidelity* (IFBench). Pass@1, ±<1% CI.

### 2.2 Broadly-used benchmarks, the capability each isolates, and the
real-world proxy

| Benchmark | Isolates | Real-world proxy | How models differentiate (2026) |
|---|---|---|---|
| **SWE-bench Verified / Pro** | repo-level multi-file code editing to a passing test | autonomous coding agents | Top cluster ~80–89% Verified (GPT-5.5 88.7); Pro still <60% — the hard frontier |
| **Terminal-Bench 2.0** | long-horizon shell agency: plan→iterate→tool-coordinate, deterministic check | CLI/devops automation agents | Wide spread: GPT-5.5 82.7 vs Qwen3.6 51.5 — strong separator |
| **τ-bench / τ²-bench (Telecom)** | multi-turn **tool use under a strict policy/API**, dialog state | customer-service / API agents | Near-saturating at the top (GPT-5.5 98.0) but mid-tier collapses — fidelity cliff |
| **BFCL V4** | function-calling: parallel/multiple selection, **relevance (when NOT to call)**, multi-turn, agentic memory | tool/function-calling fidelity | AST-graded; single-turn solved, multi-turn/relevance still discriminating |
| **GAIA** | compound multi-step tool-use + reasoning across web/files | general assistant agents | Multi-hop chains expose planning/credit-assignment failures |
| **OSWorld** | full GUI computer control, 15- vs 50-step tasks | computer-use agents | Long-task tracks isolate horizon length specifically |
| **ARC-AGI-2** | fluid abstract pattern induction, anti-memorization | genuine generalization | Largest 2026 deltas; GPT-5.5 85.0 — the prestige reasoning metric |
| **GPQA / AIME / MMLU-Pro / HLE** | hard knowledge + math reasoning | expert QA | Largely saturating top-end except HLE — *not* a transfer target for game RL (lmgame-Bench) |
| **ERQA** | **embodied spatial / trajectory / state-estimation / multi-view / task reasoning** (400 MCQ VQA) | robotics perception+planning | CoT moves it only +4–6.5pts → *reasoning-shaping* finetune is what moves it; the validated OpenRA-Bench transfer target |
| **PlanBench / Blocksworld** | precondition-ordered sequential planning, cost-optimal, replanning | task-graph scheduling, robot task planning | Mystery-Blocksworld defeats memorization; *the* planning correlate |
| **lmgame-Bench (Sokoban/Tetris RL → transfer)** | game-RL → spatial/planning transfer | the OpenRA-Bench hypothesis itself | RL→Blocksworld +≥10, →WebShop +6; NOT →GSM8K/coding |
| **SMAC → SMACv2** | multi-agent micro; the anti-memorization cautionary tale | strategy-game RL | Open-loop policy beat SMAC ⇒ procedural variation mandatory (the suite follows this) |

Net 2026 picture: differentiation has moved off knowledge/math QA
(saturating) onto **agentic tool-use fidelity, long-horizon execution,
computer/embodied control, and fluid reasoning (ARC-AGI-2)**.

---

## STEP 3 — Capability ↔ Coverage ↔ Parallel Benchmark Gap Table

| Capability / real-world use case | OpenRA-Bench coverage | Parallel benchmark | Verdict |
|---|---|---|---|
| Spatial state-estimation / perception under partial obs | `perception-target-vs-fog`, `perception-frontier-reading`, C2, C7, C8 | **ERQA** (spatial/state-est), SpatialVLM, VSR | **Well-covered** — and this is the empirically-validated transfer target |
| Frontier exploration: which unknown region to commit | `reasoning-frontier-commit`, C1 | Active Neural SLAM, frontier exploration, ERQA-trajectory | **Well-covered** |
| Constrained sequential planning (precondition graph + deadline) | C3, C4, `building-and-planning` | **PlanBench / Blocksworld**, ALFWorld | **Adequate** (parallels exist; single-map limits variety) |
| Resource allocation / multi-objective under a budget | `economy-investment`, `economy-time-box`, C5, C6 | PlanBench cost-optimal, SmartPlay | **Adequate** (spend-only) — harvest variants **non-discriminating** (engine S0/S1) |
| Risk assessment / replanning under partial info | `reasoning-risk-route`, `strategy-dilemma`, C9, C12 | PlanBench replanning, StarCraft-II-Arena | **Adequate** |
| Long-horizon multi-step credit assignment | thin: most clocks are short single-phase; C12 only gates with `after_ticks` | **Terminal-Bench 2.0**, OSWorld 50-step, GAIA | **Weak** — no genuine opening→tech→assault chain |
| Multi-unit / multi-agent coordination | `action-multiunit-coordination`, `strategy-twobody`, C10 | Watch-And-Help, SMAC(v2) | **Adequate** |
| Tool-use / function-calling **fidelity under a strict action API** | only as a *cross-cutting score* (`actions_warned/issued`); no scenario isolates it as the objective | **BFCL V4**, **τ²-bench**, IFBench | **Weak/structural gap** — yet this is the most leaderboard-weighted 2026 axis |
| Instruction-following under strict constraints | implicit in win predicates; not isolated | **IFBench** (in AA Index v4) | **Weak** |
| Adversarial / game-theoretic reasoning (opponent modeling, deception, counter-strategy) | **none** — no live adversary; `rush-hour` enemy is scripted; 1v1/Elo is Phase-1 unbuilt | StarCraft-II-Arena, game-theory evals; *the unique RTS value prop* | **Missing** — largest strategic gap |
| Tempo / timing-window discipline | C11 | TextStarCraft II, SmartPlay | **Adequate** (but only 1 category, 15 levels) |
| Fluid/abstract anti-memorization generalization | enforced by procedural variation + held-out seeds + generalization-gap metric | **ARC-AGI-2** philosophy | **Methodologically well-covered** (design discipline is correct) |

### Scenarios with weak / no transfer story (flag)

- **`rush-hour` / `strategy-gauntlet` "sweep-and-clear"** — primarily a
  search-and-destroy execution task. Defensible as multi-robot
  patrol-and-clear, but with a *scripted* enemy it tests execution,
  not adversarial reasoning. Keep, but it should not be load-bearing
  for the "strategy" claim.
- **`economy-harvest-timebox` / `economy-harvest-investment`** —
  currently economically identical to the non-harvest economy packs
  (income=0). Honest, but they are *placeholders*, not distinct tests;
  do not count them as economy coverage until S0/S1 land.
- **`cat-c5/c6/c7/c8` over-generation** — many variants differ only in
  region coordinates / target counts. The *decision* is the same; this
  is level inflation, not capability breadth (mild
  SMAC-style memorization risk if seeds aren't truly held out per
  variant). Not "game trivia," but low marginal information.
- The suite has **no** scenario that is pure game-trivia with zero
  transfer story — the `real_world_meaning`/`robotics_analogue`
  discipline is doing its job. The problem is *skew and redundancy*,
  not arbitrariness.

---

## STEP 4 — Concrete, Prioritized Edit Advice

### P0 — Add an adversarial / opponent-modeling family (highest leverage)

An RTS engine's unique, defensible value vs every other benchmark is
**live adversarial / game-theoretic reasoning** — and the suite
currently has *zero*. Build the Phase-1 second RL-controlled slot
(`EVAL_STACK_PLAN.md` Phase 1) and add:

- **Counter-Strategy Read**: opponent commits an observable opening
  (rush vs tech vs expand); agent must read it from partial recon and
  pick the dominant counter. Capability: adversarial reasoning +
  perception. Real-world: competitive multi-agent / negotiation /
  red-teaming. Parallel: **StarCraft-II-Arena (ICLR'25)**,
  game-theoretic LLM evals.
- **Deception / feint handling**: opponent shows a decoy force; win
  predicate rewards committing against the *real* axis
  (`building_in_region`/`units_killed_gte` keyed off the true threat).
  Parallel: adversarial robustness, opponent modeling.

This also makes the **Elo leaderboard** genuinely meaningful
(head-to-head), which it currently is not.

### P0 — Add a strict-action-API instruction-following family

The most leaderboard-weighted 2026 axis (τ²-bench in AA Index v4,
BFCL V4, IFBench) is only measured as a *side diagnostic* here. Add a
family whose **primary objective is action-API fidelity**:

- Tasks where the win predicate is only reachable via a *specific
  command sequence/format under explicit policy constraints* (e.g.
  "achieve X but never issue `attack` before tick T", "use only
  `move_units` + `deploy`, no `build`"), scoring relevance (issuing a
  disallowed call = fail, à la BFCL "relevance detection").
- Capability: instruction-following + tool-call fidelity under a
  strict API. Real-world: agentic API/tool orchestration, policy
  compliance. Parallel: **BFCL V4 (relevance/multi-turn), τ²-bench,
  IFBench**.

### P1 — Add 2–3 genuinely long-horizon, multi-phase scenarios

Current clocks are mostly short single-phase. Add scenarios spanning
opening→scout→economy→tech→army→engagement in **one episode (40k+
ticks)** with a terminal objective, so the score depends on **credit
assignment across phases** (an early scouting error must surface as a
late failure). Capability: long-horizon credit assignment. Parallel:
**Terminal-Bench 2.0, OSWorld 50-step track, GAIA** multi-hop. This is
where 2026 models separate most and the suite is currently thin.

### P1 — Cut / merge the redundant `cat-*` over-generation

Reduce each generated category to **≤3 levels per genuinely distinct
decision** instead of 5–6 near-duplicate coordinate variants. Net
~200→~90 levels but *higher distinct-capability density* and lower
memorization risk. Specifically merge: C5↔C6 (both "budget→units+
buildings" ECON), C7↔C8 (both "place building in inferred region"
PERC). Keep the procedural seed variation *within* a level (the
anti-memorization mechanism) — cut the redundant *level* multiplication.

### P1 — Quarantine non-discriminating economy packs

Tag `economy-harvest-*` (and any harvest variant) as **engine-prereq /
not-scored** until S0 (ore source) + S1 (silo storage) land
(`SCENARIO_BRAINSTORM.md`). Do not count them in economy coverage
claims. Land S0 (a one-line `VALID_ACTOR_TYPES` add or a
`resource_fields` scenario field) — it is cheap and unblocks a real,
distinct economy-throughput capability.

### Over/under-representation summary

- **Over-represented:** ECON spend-allocation (C5/C6 + 5 hand-authored
  economy packs, several non-discriminating), and generated PERC
  region-placement (C7/C8). Level count >> capability count.
- **Under-represented:** adversarial/game-theoretic (zero),
  tool-use/API-fidelity isolation (zero dedicated), long-horizon
  credit assignment (thin), tempo (1 category).
- **Right-sized:** spatial perception/frontier (the validated transfer
  target — keep strong), coordination, risk/replan.

### Strengthening the generalization-transfer argument

The headline claim ("rush-hour finetune lifted ERQA") needs a
pre-registered external transfer panel, scored as **per-axis deltas,
not aggregate** (lmgame-Bench protocol):

1. **ERQA** — the *primary* correlate. It is reasoning-sensitive
   (CoT-only moves it just +4–6.5pts), embodied/spatial, and is the
   axis the observed transfer hit. Report ERQA spatial / trajectory /
   task subscores separately. Strongest evidence for the claim.
2. **Blocksworld / PlanBench (incl. Mystery-Blocksworld)** —
   validates the PLAN/TECH/ECON families; this is the exact transfer
   lmgame-Bench demonstrated (RL→Blocksworld +≥10pts) and defeats
   memorization.
3. **BFCL V4 / τ²-bench** — validates the *new* P0 action-fidelity
   family and tests whether strict-action-API discipline transfers
   (the most commercially relevant axis; currently untested here).
4. **Negative controls: GSM8K + a coding eval (e.g. LiveCodeBench)** —
   lmgame-Bench showed game-RL does **not** transfer to math/coding;
   include these to demonstrate the transfer is *specific* (spatial/
   planning), not a generic capability bump. This negative result is
   what makes the positive claim credible.
5. Continue reporting the **generalization gap on held-out seeds**
   (Procgen/SMACv2/ARC-AGI discipline) — already designed in; keep it
   front-and-center as the anti-memorization guarantee.

---

## STEP 5 — Sources (verified via search/fetch, 2026-05)

- Qwen3.6-35B-A3B model card — https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Kimi-K2.6 model card — https://huggingface.co/moonshotai/Kimi-K2.6
- Introducing GPT-5.5 — https://openai.com/index/introducing-gpt-5-5/ (direct fetch 403; corroborated via search: interestingengineering.com/ai-robotics/opanai-gpt-5-5-agentic-coding-gains, kingy.ai GPT-5.5 benchmarks, llm-stats.com/models/gpt-5.5)
- Artificial Analysis leaderboard — https://artificialanalysis.ai/leaderboards/models ; methodology — https://artificialanalysis.ai/methodology/intelligence-benchmarking ; Intelligence Index — https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index
- BFCL V4 — https://gorilla.cs.berkeley.edu/leaderboard.html ; paper https://openreview.net/forum?id=2GmDdhBdDk
- τ²-bench (Sierra) via Agentic AI Benchmarks — https://awesomeagents.ai/leaderboards/agentic-ai-benchmarks-leaderboard/
- ERQA — https://github.com/embodiedreasoning/ERQA ; Gemini Robotics arXiv:2503.20020 https://arxiv.org/html/2503.20020v1
- lmgame-Bench arXiv:2505.15146 — https://arxiv.org/abs/2505.15146
- GAIA / OSWorld / Terminal-Bench / ARC-AGI-2 overview — https://www.marktechpost.com/2026/04/26/top-7-benchmarks-that-actually-matter-for-agentic-reasoning-in-large-language-models/ ; https://www.spheron.network/blog/ai-agent-benchmarking-gpu-cloud-swebench-gaia/
- In-repo: `SCENARIO_CATALOG.md`, `SCENARIO_BRAINSTORM.md`, `EVAL_STACK_PLAN.md`, `openra_bench/scenarios/win_conditions.py`, `openra_bench/scoring.py`, `openra_bench/goal_tracker.py`, `openra_bench/scenarios/packs/*.yaml`
- Supporting literature cited in-repo and corroborated: SMAC→SMACv2 (arXiv:2212.07489), Procgen (arXiv:1912.01588), ARC-AGI (arXiv:1911.01547), PlanBench (arXiv:2206.10498), Active Neural SLAM (arXiv:2004.05155)
