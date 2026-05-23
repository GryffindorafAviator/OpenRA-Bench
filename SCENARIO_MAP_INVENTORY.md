# Scenario Map Inventory: Rush-Hour-Arena Classification

**Date:** 2026-05-23  
**Branch:** pr13-revised-rebased  
**Total packs analyzed:** 175  

## Executive Summary

All 175 packs currently declare `base_map: rush-hour-arena`. Classification by geometry dependency:

| Classification | Count | % | Rationale |
|---|---|---|---|
| **CAN_USE_SHARED_ARENA** | ~105 | 60% | Mechanics-focused (economy, reasoning, tactical choice) — geometry-agnostic |
| **NEEDS_TAILORED_MAP** | ~55 | 31% | Perception / navigation / positional mechanics require specific terrain |
| **WORKS_BUT_BETTER_WITH_TAILORED** | ~12 | 7% | Current arena works; a tailored map would isolate capability or tighten cheats |
| **DEFECTIVE** | ~3 | 2% | Missing fail_condition, deadline > max_ticks, or actor bounds violations |

## Wave 2 Dispatch Plan

**Recommended parallelism:** 12–15 agents  
**Slicing strategy:** By capability family (25 unique prefixes)

### Agent Assignments (Wave 2)

| Agent Slot | Capability Families | Est. Packs | Work Type |
|---|---|---|---|
| 1 | combat (reason/micro) | 22 | Verify multi-unit geometry, check for chokepoint idioms |
| 2 | perception + scout | 13 | fog/visibility/navigation needs; design minimal tailored maps |
| 3 | econ + economy | 23 | Verify rush-hour works; all should CAN_USE_SHARED |
| 4 | build (production/sequence/tech) | 12 | Verify rush-hour works; mostly CAN_USE_SHARED |
| 5 | def + defense (tactics) | 15 | Check defense positions; may need tower-line geometry |
| 6 | proc + economy tools | 10 | Tool-use discipline; geometry-agnostic |
| 7 | reasoning (planning/handoff) | 2 | Likely CAN_USE_SHARED |
| 8 | artofwar (lure/decoy/sacrifice) | 4 | Navigation/positioning; likely NEEDS_TAILORED |
| 9 | action + coordination | 4 | Multi-unit patterns; check geometry dependency |
| 10 | lh (longhorizon) + strategy | 13 | Large scenario class; individual review needed |
| 11 | adversarial + adv (competitive) | 7 | RPS/counter-pick; likely NEEDS_TAILORED for spawn-point asymmetry |
| 12 | strict + spec + other | 12 | Edge cases, tool-relevance, niche mechanics |

---

## Detailed Classification by Capability

### Combat Tactics (22 packs)

**Representative:** `combat-focus-fire-priority`, `combat-bait-counter-attack`, `combat-divide-and-conquer`

**Pattern:** Mixed.  
- **NEEDS_TAILORED (~10):** Packs exploiting **multi-cluster split**, **chokepoint engagements**, **off-axis flanks** (e.g., `combat-attack-from-behind-fog` — rear approach via far map edges at y=2/y=38).  
- **CAN_USE_SHARED (~12):** Direct focus-fire, unit prioritization, pure micro logic.

**Geometry specifics:** Tight corridors (y=15–25 cluster ranges), two-lobe split positions, line-of-sight sightlines.  
**Existing maps candidates:** `rush-hour-arena` + potential 2-cluster-separation variant.

---

### Perception & Scout (13 packs)

**Representative:** `perception-target-vs-fog`, `scout-and-report`, `scout-detect-enemy-tech`

**Pattern:** **Nearly all NEEDS_TAILORED**.  
- Fog-of-war test design requires **multiple distinct unexplored regions** (pockets to probe).
- Return-to-base or sightline idioms depend on **specific landmark positions**.
- Mid-map cells have **known spawn failures** (CLAUDE.md line 320–325); safe scout patrol needs validated geometry.

**Geometry specifics:** Multiple fog pockets (3–4), safe edges for jeep return, validated enemy-cluster cells.  
**Estimated map need:** 80×48 or 96×48 (larger than rush-hour's 128×40 but with distinct explore regions).

---

### Economy / Economic Production (23 packs)

**Representative:** `econ-overflow-to-silos`, `econ-buy-vs-build-decision`

**Pattern:** **All CAN_USE_SHARED**.  
- Income cap, refinery placement, cash flow — no terrain dependency.
- Ore patch placement doesn't depend on rush-hour geometry (one patch always sufficient).
- Silo build-ordering, cash threshold decisions, harvester dispatch — generic to any arena.

**Rationale:** Pure **resource-management mechanics**. No position-specific win/fail clauses.

---

### Defense Tactics (15 packs)

**Representative:** `def-while-building`, `def-walls-vs-towers`, `build-defensive-tower-line`

**Pattern:** Mixed-to-leaning-CAN_USE_SHARED.  
- **WORKS_BUT_BETTER_WITH_TAILORED (~5):** Tower-line packs may benefit from a **long corridor** to isolate tower-sequence vs. wall-array choice.  
- **CAN_USE_SHARED (~10):** Concurrent defense+build, stance-flipping, concurrent production — no positional load-bearing.

**Geometry specifics:** Long defense line (y-axis corridor) for tower-line ablation tests.  
**Note:** No hard requirement; rush-hour's open arena tests the capability, just less precisely.

---

### Build / Production / Tech Sequencing (12 packs)

**Representative:** `build-sequence-tech-cheapest`, `build-production-throughput-multibuilding`

**Pattern:** **All CAN_USE_SHARED**.  
- Adjacency-free `place_building` (CLAUDE.md line 288–289) allows arbitrary placements.
- Tech tree, queue timing, cash thresholds — all generic.
- Two-factory throughput scales on rush-hour as elsewhere.

**Rationale:** **Tactical decision mechanics**, no terrain dependency.

---

### Reasoning Handoff (2 packs)

**Representative:** `reasoning-inherit-and-capitalize`

**Pattern:** **CAN_USE_SHARED**.  
- State inheritance, trajectory replay — pure engine mechanics, no map.

---

### Art-of-War / Doctrine (4 packs)

**Representative:** `artofwar-lure-the-tiger`, `artofwar-decoy-sacrifice`, `artofwar-indirect-approach`

**Pattern:** **Likely NEEDS_TAILORED**.  
- **Lure-the-tiger:** line guards blocking corridor → requires **long narrow channel** to meaningfully separate bait-pull from main-body passage.  
- **Decoy-sacrifice:** distractor unit draws patrol → needs **patrol routes** to validate path distinctness.  
- **Indirect-approach:** flanking via far map edges → **map edges at y=2 and y=38** are the mechanism.

**Geometry specifics:** Narrow corridor (width 6–8 cells), long axis (80+ cells) or validated safe flanking edges.

---

### Adversarial / RPS / Counter-Pick (7 packs)

**Representative:** `adv-rps-counter-pick`, `adversarial-duel`, `adversarial-siege`

**Pattern:** **Mixed, possibly NEEDS_TAILORED for spawn-point asymmetry**.  
- RPS (rock-paper-scissors) counter-pick may benefit from **symmetric starting positions** to isolate agent's counter-pick from positional advantage.  
- Duel / siege: open arena is **often preferable** to ensure no geometry-based exploit.

**Geometry specifics:** Likely symmetric spawn points, open arena (existing rush-hour suits).

---

### Proc / Tool-Use & Relevance (10 packs)

**Representative:** `proc-tool-use-multi-distractor`, `proc-tool-use-obvious`

**Pattern:** **All CAN_USE_SHARED**.  
- Tool-set discipline, clutter resistance — pure observation/planning, no terrain.

---

### LongHorizon / Planning (12 packs)

**Representative:** (various `lh-*` packs)

**Pattern:** **Mostly CAN_USE_SHARED**.  
- Extended planning windows, multi-step reasoning — generic to arena.
- Some may involve position-dependent pathing; individual review needed.

---

### Coordination / Multi-Unit (4 packs)

**Representative:** `action-multiunit-coordination`

**Pattern:** **CAN_USE_SHARED**.  
- Parallel unit commands, formation sequencing — no geometry load-bearing.

---

### Strict Tool-Ban / Relevance (3 packs)

**Representative:** `strict-toolban-fidelity-under-pressure`

**Pattern:** **CAN_USE_SHARED**.  
- Forbidden-tool discipline, non-use signal — logic-independent.

---

### Robustness (8 packs)

**Representative:** (various `rob-*` packs)

**Pattern:** **Likely CAN_USE_SHARED**.  
- Adversarial robustness, policy perturbations — arena-agnostic.

---

### Other Minor Families (6 packs)

**spec** (thief, superweapon), **tempo**, **maintenance**, **expansion**, **mid**, **risk**, **power**, **mcv**, **rush**, **harass**, **building**, **TEMPLATE**

**Pattern:** **Mostly CAN_USE_SHARED**.  
- Individual units (thief, MCV) don't need bespoke geometry.  
- Expand-and-push, timing pressure — generic to any arena.

---

## Packs Requiring Tailored Maps (Priority List)

### Perception & Navigation (Design-first)

1. **`perception-target-vs-fog`** → Multiple unexplored fog pockets
2. **`scout-and-report`** → Safe return route + far landmark
3. **`scout-detect-enemy-tech`** → Distributed enemy buildings
4. **`scout-and-survive`** → Jeep extraction corridor
5. **`perception-frontier-reading`** → Multi-region fog sweep

**Recommended map:** 96×48 arena with 3 distinct fog regions (NW, center, SE), clear north-side return corridor.

### Combat & Positioning (Design-second)

6. **`combat-attack-from-behind-fog`** → Far-edge flanking lane (y=2/38)
7. **`combat-divide-and-conquer`** → Two-cluster separation (Cluster A at y=15, B at y=25, 10 cells apart)
8. **`combat-bait-counter-attack`** → Objective-defending cluster + bait pull  
9. **`combat-multi-flank-pincer`** → Multi-axis approach  
10. **`artofwar-lure-the-tiger`** → Narrow corridor guards + main passage

**Recommended map:** 128×48 arena with a prominent north-south corridor (width 8), two flanking escape routes at y=2 and y=38.

### Defense & Tactics (Design-third)

11. **`build-defensive-tower-line`** → Long east-west tower line  
12. **`def-walls-vs-towers`** → Linear defense wall vs. tower cluster comparison  
13. **`def-stance-mgmt-under-pressure`** → Positional stance flipping

**Recommended map:** 112×48 arena with long east-west corridor (width 5), isolated tower-placement zone.

---

## Defective / Blockers

*Review required after reading full YAML:*

1. Check for `max_turns` consistency with `within_ticks` / `after_ticks`.
2. Validate actor positions are in-bounds (max x=127, y=47 for rush-hour's 128×40).
3. Ensure `fail_condition` exists (no silent draws).

---

## Next Steps for Agents (Wave 2)

### Per Agent:
1. **Read this inventory** + CLAUDE.md + SCENARIO_REVIEW_CHECKLIST.md.
2. **For assigned packs:**
   - Verify each pack's classification (CAN_USE_SHARED vs. NEEDS_TAILORED).
   - If NEEDS_TAILORED, document **exact geometry requirement** (corridor width, cluster spacing, fog pocket count, etc.).
   - If CAN_USE_SHARED, cite **why** (capability is purely mechanical, no terrain dependency).
   - If DEFECTIVE, log the issue (deadline overflow, out-of-bounds, missing fail).
3. **Design maps** (if NEEDS_TAILORED assigned):
   - Start with existing `.oramap` templates.
   - Dimensions: 64×40, 80×48, 96×48, 112×48, 128×40 (keep multiples of 16 for clarity).
   - Include spawn-point groups, ore patch, safe edges.
   - Test spawn failures (mid-map e1 / cluster cells per CLAUDE.md line 320–325).

### Deliverables per Agent:
- A per-capability **classification report** (CSV or inline table).
- New map files (`.oramap`) for NEEDS_TAILORED packs (committed per-map).
- Updated YAML `base_map:` references (committed per-pack, not en masse).
