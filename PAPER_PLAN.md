# OpenRA-Bench — Paper Plan & Critique

Working document. Captures the central thesis, the paper framing, the
three findings, the experiment program, and the open critiques /
reviewer-defenses. Living doc — update as experiments land.

---

## 1. Central thesis (as stated)

We build a benchmark on the Real-Time Strategy game *Red Alert* and
measure LLM performance on **adversarial, multi-modal, long-horizon
strategic planning and execution**. Three findings:

1. **The major gap is image-based perception.** In pure-text
   battlefields models execute complex strategies; in mixed modality
   — where own and enemy positions must be read off a minimap — they
   struggle. We show this by SFT on a small set of text-modality
   results with the image attached, and observe improvement on this
   benchmark *and* on external vision benchmarks (ERQA).
2. **Models panic in unfavorable / uncertain situations.** Scores
   drop sharply under fog of war while humans hold a baseline. Under
   heavy losses, models choose `observe`/`stop` instead of actively
   redirecting units. Hypothesis: a transfer of behavior from
   code-heavy training, where the rewarded move in an unfavorable
   state is to stop and wait for human intervention.
3. **Bridging both gaps via simple SFT** lets a *smaller* model beat
   large reasoning models on 1v1 ELO.

Contributions: 200 research-grounded scenarios; an active 1v1
battleground; evidence that models form complex, diverse strategies
despite the perception and adversity gaps (and that strategy varies
with temperature and is model-characteristic — turtle / balanced /
aggressive).

---

## 2. Framing & positioning

**The hook is the inversion:** LLMs can do the *hard* part
(long-horizon adversarial strategy) but fail the "easy" parts
(reading a map, holding their nerve). Lead with that — it inverts the
usual "LLMs can't plan" narrative, and the failures are specific and
*fixable*.

Position OpenRA-Bench as a **diagnostic instrument**, not a
leaderboard. The methodological novelty vs. the crowded
SC2 / TextStarCraft LLM-agent space is the **ablation methodology that
decomposes failure** into perception / reasoning / action / adversity.
Arc of the paper: **diagnosis → localization → treatment → transfer.**

Title candidates:
- *Map-Blind and Panic-Prone: Diagnosing LLM Strategic Agents in
  Real-Time Strategy*
- *The Bottleneck Is Perception, Not Planning: A Diagnostic RTS
  Benchmark*
- *LLMs Can Strategize But Cannot See: Decomposing Agent Failure in
  Red Alert*

Venue: strong enough for a main track (ICLR/NeurIPS/ICML) on the
diagnosis+treatment+transfer arc — not only Datasets & Benchmarks.

---

## 3. Contributions (sharpened)

1. **OpenRA-Bench** — 200 controlled RTS scenarios on a deterministic
   engine, each anchored to a named real-world capability / external
   benchmark, built to a **no-cheat / no-defect** bar (every lazy /
   brute / stall policy provably loses on every level and seed). The
   validation rigor is itself a contribution: it is why the benchmark
   measures the *intended* capability.
2. **An ablation methodology that decomposes failure** — the
   perception grid (channel × fog) and the handoff sweep — turning
   "the model lost" into "the model lost *because of perception /
   adversity / planning*."
3. **A 1v1 self-play battleground** with ELO.
4. **Two localized, fixable gaps** — perception-binding and
   inaction-under-adversity — each with an SFT remedy, cross-benchmark
   transfer (ERQA), and a small-beats-large result.

---

## 4. Paper structure

§1 Intro (the inversion) · §2 Related work · §3 OpenRA-Bench (engine,
200 scenarios, no-cheat design, ablation axes, 1v1/ELO,
human-labeling) · §4 Setup (models, sweeps, metrics) · **§5 Models
*can* strategize** (the diversity control) · §6 Finding 1: perception
gap · §7 Finding 2: panic · §8 Finding 3: SFT remedy + transfer ·
§9 Limitations · §10 Conclusion.

**Key structural move:** promote the "diverse strategies" result from
a bonus to **§5, before the failure findings.** Showing models produce
coherent, diverse, model-characteristic strategies is what licenses
the claim that the later failures are *not* failures of strategic
reasoning. Without it a reviewer says "maybe they just can't plan."

---

## 5. §5 control — models *can* strategize

Promote from "bonus" to a load-bearing control.

- Classify each game's strategy (turtle / balanced / aggressive) via a
  trajectory classifier (first-attack tick, army-vs-econ ratio,
  expansion count, aggression index, build order). Show per-model
  distributions differ.
- Temperature sweep: strategy entropy vs. temperature for one model.
- Show strategies are *coherent* (executed consistently once chosen),
  not random.

**The headline figure — the strategy-embedding scatter, in 1v1.** Each
game's trajectory is featurized and embedded to 2D (PCA / UMAP);
points colored by model. The plot shows, at a glance:
- **inter-model clusters** — model A lives in the aggressive region,
  model B turtles — models have characteristic strategy *priors*;
- **intra-model spread** — the same model across games (especially at
  high temperature) scatters — one model generates *diverse* play.
1v1 is the right venue: full-macro adversarial games make strategy
legible in a way single-scenario tasks do not. A bigger model roster
(8–12, Together + Bedrock) makes the clustering visually striking.

This is the evidence that planning is not the bottleneck — and a
genuinely compelling standalone result.

---

## 6. Finding 1 — Perception gap

**Claim, sharpened.** Decompose perception into **extraction** (read
state from the image) and **binding** (act on image-derived state).
The perception sweep already separates these: `image`-channel on
perception packs (count / locate) isolates extraction; `image` on
action packs isolates binding. Report *which* breaks. The result is
"the perception→action **binding** fails while pure planning and pure
extraction are intact" — not the weak "mixed modality is hard."

**Experiments.**
- Perception sweep (6 cells) × model roster × scenario set × seeds →
  the modality-gap number per model (`structured − image`).
- A "perceive-then-act" scaffold (force the model to transcribe
  positions first) — if it rescues performance, the bottleneck is
  binding, not extraction.
- **Cross-modal action distillation SFT**: text-mode *winning*
  trajectories + the attached minimap → train image→action. Eval
  `image` channel on **held-out** scenarios.
- **Transfer**: ERQA + ≥2 other spatial / vision benchmarks
  (e.g. BLINK, a spatial-VQA set), before / after SFT.

**Why transfer is the crown jewel.** Training image→action where the
action is good could teach good *action priors* without the model
ever using the image. The cross-benchmark transfer is what proves the
SFT taught the model to *see*. Without it, finding 1 is "task
finetuning helps task." With it, it is "agentic RTS data improves
general visual spatial reasoning." Use ≥2 external benchmarks so it
cannot be called cherry-picked.

**Reviewer attack → defense.** "VLMs are known to be bad at dense
small images — this is just OCR." → We measure it in a *consequential
agentic loop*, we *localize* extraction vs binding, and we show a fix
that *transfers*. Plus a render-style mini-ablation (the
`constant_colors` / scale knobs) so the gap is not an artifact of one
minimap style.

---

## 7. Finding 2 — Panic / inaction under adversity

**Claim.** Models degrade far more than humans under fog; under heavy
losses they default to `observe` / `stop` instead of active redirect.
Operational definition = the **`passivity` metric** (fraction of the
model's turns spent on `observe` / `stop` only). Keep "panic" only as
an informal label; formal text says "inaction bias under adversity."

**Experiments.**
- Fog ablation × models × **humans** — report the *fog-penalty gap*
  (human degrades little, model degrades a lot), not raw fog scores.
- Handoff bad-prefix → passivity, models vs. human. Models freeze
  (high passivity), humans redirect (low).
- Show passivity is *causally costly* — within-model, passive turns
  predict worse outcomes controlling for position quality.

**The coding-bias hypothesis is the spiciest claim — handle with
care.** Cannot be asserted as fact. Keep as a hypothesis AND support
with ≥1 of:
- **base vs. RLHF / instruct** versions of the same model — if base
  models panic less, post-training is implicated;
- a **loss-aware prompt intervention** ("inaction is costly; never
  just observe when losing") — if a prompt largely fixes it, the
  behavior is a shallow prior, consistent with the hypothesis;
- passivity vs. known code / RLHF intensity across the roster.

**Critical-path risk.** The human baseline needs *real human data at
scale*. The Play tab is built; data is not collected. Scope: a
representative scenario subset, several humans, disclosed RTS skill. A
strong scripted policy can serve as a secondary "competent
non-panicking" reference if human N is thin.

---

## 8. Finding 3 — SFT remedy, small beats large

**Reviewer attack (serious).** "A task-finetuned small model beating a
zero-shot large model is trivial." **Defenses to build in:**
- The SFT is **small and targeted** (perception grounding + active
  recovery), explicitly *not* trained on eval scenarios — state the
  train / eval split loudly.
- **Also SFT the large model** — show the gap is real and fixable at
  every scale. Headline becomes: "the perception+panic gap is large
  enough that closing it in a small model *outweighs a 10× reasoning
  advantage* — and it was never a reasoning gap."
- Ablate the SFT: perception-only / recovery-only / both — each
  component contributes.
- ELO with enough games + confidence intervals; held-out 1v1 maps.

**Infrastructure synergy (state explicitly).** The handoff
`TrajectoryController` + the human-Playback format *are* the SFT data
pipeline — recovered bad-prefix episodes are active-recovery
exemplars; text-mode wins are perception-distillation data. The
ablation infrastructure doubles as the training-data factory.

---

## 9. Experiment program

| # | Experiment | Infra | Status |
|---|---|---|---|
| §5 strategy diversity | classify 1v1 games; temp sweep; per-model dists | 1v1 harness ✓; needs strategy classifier | TODO |
| Perception sweep | 6-cell × roster × scenarios × seeds | ✓ `--perception-sweep` | **run** |
| Handoff / passivity | base/bad/good × roster | ✓ `--handoff-sweep` | **run** |
| Human baseline | fog × scenario subset × humans | Play tab ✓, **no data** | **highest-risk** |
| Cross-modal SFT + transfer | distill text→image; ERQA + 2 more | data pipeline ✓, needs finetuning | **biggest compute** |
| 1v1 ELO tournament | round-robin + CIs | harness ✓ | run |
| Recovery SFT | active-recovery exemplars → finetune | handoff bank ✓ | run |

Critical path: human baseline (logistics) and the SFT (compute).

---

## 10. Metrics

Win-rate / outcome · composite P/R/A score · objective-progress
(continuous) · ELO (1v1, with CIs) · **passivity** (freeze metric) ·
generalization gap (public vs. held-out seeds) · strategy class /
entropy · human-normalized score (model / human) · derived gaps:
modality gap = `score(structured) − score(image)`, fog penalty =
`score(clear) − score(fog)`, per model and per human.

---

## 11. Threats to validity / limitations to preempt

- **One game (RA).** Lean on the capability taxonomy
  (`meta.benchmark_anchor`) + the ERQA transfer for generality.
- **Engine is a reimplementation.** Deterministic + validated is the
  answer.
- **One minimap render style.** Render-robustness ablation.
- **Human skill / N.** Disclose; representative subset.
- **"Panic = code training."** Hypothesis, not claim — support with
  the probes in §7.
- **SFT leakage.** Loud train/eval scenario split.
- **ELO methodology.** Game count, pairing, confidence intervals.

---

## 12. Pre-full-run audits (must land before the 200-pack sweep)

After the pilot finishes and *before* committing compute to the full
200-pack run, three audits gate the rigor of the headline numbers:

### 12.1 Scenario quality audit
Two layers:
- **Static** — re-run the scripted-policy bar (`stall` / `brute` /
  `intended`) across all 200 packs. Engine fixes may have drifted a
  pack since authoring (a lazy policy now wins, or `intended` now
  loses). Catches benchmark rot.
- **Empirical** — from pilot/full-run data, flag packs where *every*
  model wins (too easy / a trivial idiom dominates — task #43) or
  *every* model loses (unsolvable or a predicate is mis-tuned —
  task #44). Discriminative packs are the only useful ones.

Paper payoff: a post-hoc audit table converts the "no-defect bar"
claim into something you can *show* — a strong methodology subsection.

### 12.2 Coverage map — RTS phase × decision-divergence
Map all 200 packs (plus the 1v1 battleground) onto the **RTS phase ×
decision-divergence matrix** from the original plan
(opening / early-mid / mid / mid-late / late × the canonical decisions
in each). Produce a coverage heatmap; flag empty / thin cells. Surface
the `meta.capability`-tag imbalance (`adversarial`=1 pack — full
end-to-end macro lives in the 1v1 battleground, both belong on the
map). Paper payoff: a figure showing the bench spans the *real* game,
not just easy probes.

### 12.3 Multi-run reliability — `pass^k`
Each (cell, seed) is run N times varying only model nondeterminism
(requires temperature > 0). Report mean ± CI **and** `pass^k`
(all-k-wins). A model that wins 5/10 identical runs is a fundamentally
different finding than 10/10. `--repeats N` in `run_eval`; default
`k=5` (Codex / SWE-bench convention). Paper payoff: mean-only is
fragile; reliability is itself a possible headline result.

## 13. Stretch ideas

- **Pivotal-turn analysis** — single-turn counterfactual swaps to show
  RTS losses are 1–2 catastrophic decisions, not uniform decay.

---

## 13. Ablation infrastructure already built (this is real, today)

- **Fog axis** — engine `reveal_map` no-fog flag (`OpenRA-Rust`),
  the `-clear` perception cells.
- **Modality axis** — `structured` / `vision` / `image` (image-
  primary, text redacted, labelled minimap) channels;
  `run_eval --perception-sweep` expands `pack:level` into the 6
  modality cells.
- **Handoff axis** — `openra_bench/handoff.py`
  (`HandoffController`, `TrajectoryController`), `run_eval
  --handoff-sweep`; the `passivity` metric on every result.
- **1v1 battleground + ELO** — `one_v_one.py`, scripted ladder.
- **Human-labeling** — the Play tab persists human runs in the
  standard `Playback` format (apples-to-apples with model runs).
- **200 scenario packs** — no-cheat-validated, capability-anchored.

---

## 14. Open decisions

- Model roster for the sweeps (which models; vision-capable required
  for `vision` / `image` channels).
- Compute / API budget for the full sweeps.
- Human-study scope (how many humans, which scenario subset).
- SFT base model(s) and the small/large pairing for finding 3.
- Strategy-classifier definition for §5.
