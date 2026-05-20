# OpenRA-Bench — Scenario Backlog (toward 200 packs)

This file is the canonical brainstorm catalog for the path from
**29 packs today** (22 active + 17 designed in the plan + 2 harvest
rebuilds) → **~200 packs total**. It complements
`SCENARIO_REVIEW_CHECKLIST.md` (methodology) and `SCENARIO_QUALITY.md`
(whole-suite recap) by giving the next-buildable seed any coding agent
can pull off.

## How to use this file

1. Pick a seed whose **engineering tag** matches what's available
   today (`YAML` = build now; `BOT:x` / `PRED:x` / `ENG:x` = blocked
   on that engine PR landing).
2. Author the pack to the **no defect, no cheat** bar (see
   `SCENARIO_REVIEW_CHECKLIST.md` and `CLAUDE.md`).
3. Populate the new schema field `meta.benchmark_anchor: list[str]`
   with the seed's anchor (real-world capability and/or named
   benchmark).
4. Scripted-policy validate (stall / brute / wrong-path / intended)
   across seeds 1–4 — every lazy policy must LOSS, intended must WIN,
   never a DRAW.
5. Move the seed from this file to a section of
   `SCENARIO_QUALITY.md` (the whole-suite catalog) once committed.

## Engineering tags

- `YAML` — buildable today, pure YAML.
- `MAP:x` — needs a new map generator or static `.oramap`.
- `BOT:x` — needs new scripted-bot variant in
  `openra-sim/src/scripted_bot.rs`. Currently supported: `hunt`,
  `rusher`, `patrol`, `turtle`, `guard`. Proposed: `raider`, `pincer`,
  `feinter`, `prober`, `tower-rusher`, `boomer`, `tech-switcher`.
- `PRED:x` — needs new predicate in
  `openra_bench/scenarios/win_conditions.py` (+ `_PHRASES` entry in
  `game_knowledge.py`).
- `ENG:x` — needs deeper engine work (new order, signal, mid-episode
  hook). Examples: `deploy`, `capture_actor`, `apc-transport-complete`,
  `air-units`, `power-signals` (today inert), `scheduled-event`,
  `dynamic-deadline`, `RL-adversary` (P3-1).

## Engineering shopping list (PRs that unblock the most seeds)

Order from cheapest to most expensive; numbers in parentheses are how
many seeds each PR unblocks.

1. **`forbidden_tools` + `tool_violation` signal** (P2-1) — unblocks
   ~6 seeds (Group I + B3).
2. **`raider` bot** — unblocks ~4 (B1 + Group F defenses + C2).
3. **`then:[A,B]` happened-before composite** (P2-3) — unblocks ~5
   (B2 + Group G/I/N).
4. **`own_building_lost_gte` + `building_destroyed_gte`** (P2-2) —
   unblocks ~3 (C3 + C2 + C5).
5. **`region_held_for_ticks`** — strengthens B4 + several defense
   seeds.
6. **`pincer` / `feinter` bots** — unblocks C3, C4, several adv-*.
7. **Mid-episode scripted-event hook** — unblocks C5 + 4 robustness
   seeds + surprise-flank.
8. **Power-signals fix** — unblocks 7+ seeds in Groups A/M.
9. **MCV `deploy` order** — unblocks ALL Group A `mcv-*` (7+ seeds).
10. **`capture_actor` order** — unblocks Group K spec-engineer-*.
11. **Engineer-rebuild order** — unblocks several robustness seeds.
12. **`apc` transport completeness** — unblocks Group L.
13. **Water mapgen + `lst`** — unblocks Group L amphibious + island.
14. **Air units (`yak`/`mig`/`heli`)** — unblocks anti-air,
    paradrop, air-ground coord.
15. **Stealth/`spy`/`thief` mechanics** — unblocks spec-spy / spec-thief.
16. **Superweapons** (nuke / iron-curtain / chronosphere) — Group K.
17. **Live RL adversary slot (P3-1)** — unblocks C1, C4, all
    adv-*-race seeds.

---

# Backlog seeds organized by family

Each seed has: `id` — cell — real-world anchor // benchmark anchor —
one-line predicate sketch — engineering tag(s).

## Group A — Base building & MCV (25)

Largely engine-gated on `deploy`; pre-`deploy` proxies via pre-placed
`fact` markers + build-radius creep.

- `mcv-deploy-and-build` — Opening MCV→fact site choice // critical-infra siting; SC2LE BO — `has_building:fact` AND deploy-tick ≤ T AND `building_total_gte:3` — ENG:deploy + YAML
- `mcv-deploy-defensible-site` — Opening site under threat // resilience-first siting; MicroRTS terrain-aware — `building_in_region:{good_terrain,fact,1}` AND survival — ENG:deploy + YAML
- `mcv-deploy-near-resource` — Opening site near ore // econ-first siting; SC2LE econ — `building_in_region:{near_ore,fact,1}` AND `economy_value_gte:N` — ENG:deploy + YAML
- `mcv-deploy-relocate-under-pressure` — Mid conyard pack + relocate // BCP relocation; PlanBench replanning — `building_in_region:{new_site,fact,1}` after `after_ticks:T_loss` — ENG:deploy+undeploy + YAML
- `mcv-deploy-second-base` — Mid 2nd MCV → 2nd base // multi-site ramp; MicroRTS expansion — 2 facts in 2 regions — ENG:deploy + YAML
- `mcv-deploy-third-base` — Mid-Late 3rd MCV → 3rd base // tri-base macro; SC2 3-base — 3 facts in 3 regions — ENG:deploy + YAML
- `mcv-deploy-evac-and-resite` — Mid evacuate failing base // disaster recovery; ScienceWorld — fact razed THEN new fact built — ENG:deploy + YAML
- `build-radius-creep-to-objective` — Mid walk base toward enemy // FOB; SC2 forward pylon — `building_in_region:{near_enemy,pbox,1}` — YAML
- `build-radius-creep-bridge-island` — Mid-Late creep over bridge // staging expansion; MicroRTS terrain — `building_in_region:{island,fact,1}` — ENG:bridge-aware-mapgen + YAML
- `build-power-online-first` — Opening `has_building:powr` before all else // grid bring-up sequencing; PlanBench ordering — `then:[powr, _]` — PRED:then + YAML
- `build-defensive-tower-line` — Mid 4 pbox in line covering choke // perimeter design; ERQA spatial — `building_count_gte:{type:pbox,n:4}` + `building_in_region:{choke,pbox,4}` — YAML
- `build-defensive-tower-cluster` — Mid pbox cluster at bottleneck // concentrated defense; ERQA — `building_in_region:{bottleneck,pbox,4}` — YAML
- `build-defensive-skirt-corners` — Mid corner-mounted defenses // distributed defense; MicroRTS — `building_in_region` 4x corner regions — YAML
- `build-spacing-anti-splash` — Mid space buildings to survive splash // resilience design; robustness — `min_separation_gte:N` — PRED:min_separation + YAML
- `build-anti-air-positioning` — Mid SAM coverage // anti-air planning; SC2 antiair — `building_in_region` of `sam` — ENG:air-units + YAML
- `build-engineer-rebuild-after-loss` — Mid engineer rebuilds razed building // resilience drill; PlanBench replanning — building re-presence after loss — ENG:engineer-rebuild-order + YAML
- `build-sequence-tech-fastest` — Opening BO hits `weap` in min turns // BOM-optimal; PlanBench cost-optimal — `then:[tent, weap]` AND `within_ticks:T_tight` — PRED:then + YAML
- `build-sequence-tech-cheapest` — Opening BO hits `weap` cheapest // BOM-cost; PlanBench cost-optimal — cash-bound — YAML
- `build-sequence-tech-most-resilient` — Mid tech via redundant prereqs // resilience; PlanBench robust — 2x prereq buildings — PRED:then + YAML
- `build-power-budget-online` — Opening keep `power_surplus_gte:0` // grid bring-up; SC2 power — ENG:power-signals + YAML
- `build-power-down-defensive` — Mid `power_down` non-essentials under attack // load-shedding; ops runbook — YAML
- `build-repair-priority-under-fire` — Mid repair structures in correct order // repair triage; DR — YAML
- `build-sell-and-rebuild-elsewhere` — Mid `sell` weakened, rebuild defended // capital reallocation; finance — YAML
- `build-rally-point-management` — Opening set rally forward // production logistics; warehouse SLA — YAML
- `build-production-throughput-multibuilding` — Mid 2x `weap` parallel queue // throughput; queueing theory — YAML

## Group B — Multi-front base building (12)

- `mfb-two-base-simultaneous` — Mid build base-2 while base-1 builds // multi-site ramp; SC2 sim-base — 2 facts within tight window — YAML
- `mfb-base-1-defend-base-2-build` — Mid defend live + build forward // ops continuity during expansion; DR siting — YAML
- `mfb-third-base-against-clock` — Mid-Late 3rd base before deadline // greedy expansion; SC2 3-base timing — YAML
- `mfb-relocate-conyard-under-pressure` — Mid pack + move conyard // BCP; PlanBench replanning — ENG:deploy/undeploy + YAML
- `mfb-evacuate-base-deploy-elsewhere` — Mid abandon base, redeploy MCV // disaster recovery; ScienceWorld — ENG:deploy + YAML
- `mfb-tech-base-vs-economy-base` — Mid-Late role-specialized bases // specialization; industrial — YAML
- `mfb-island-base-amphibious` — Late build base across water // amphibious staging; RoboMaster ship-to-shore — ENG:transport+amphib-mapgen + YAML
- `mfb-forward-base-near-enemy-corner` — Mid-Late FOB // SC2 forward production — YAML
- `mfb-redundant-tech-buildings` — Mid 2x `weap` for resilience // redundancy; robust planning — YAML
- `mfb-mirror-base-east-west` — Mid symmetric multi-base // distributed-systems mirror; raft replicas — YAML
- `mfb-rotating-production-pressure` — Mid-Late alternate production under raid // load balancing; queueing — PRED:then + YAML
- `mfb-supply-line-link-between-bases` — Mid-Late connect 2 bases via patrol // logistics route; supply-chain — YAML

## Group C — Defense positioning & surprise reaction (16)

- `def-position-expected-direction` — Mid fortify N (intel says N) // pre-positioned defense; ERQA spatial commit — YAML
- `def-position-revealed-direction` — Mid fortify after scout reveals // adaptive defense; PlanBench replanning — YAML
- `def-surprise-flank-react` — Mid intel said N, enemy comes S // surprise reaction; adversarial robustness — ENG:enemy-spawn-misdirection + YAML
- `def-pre-position-mobile-reserve` — Mid mobile reserve in center // reserve doctrine; military — YAML
- `def-tower-line-vs-cluster` — Mid choose spread vs concentration // defense topology; graph min-cut — YAML
- `def-walls-vs-towers` — Mid passive vs active mitigation // security architecture — ENG:walls + YAML
- `def-engineer-repair-under-fire` — Mid which tower to repair under fire // triage; DR — YAML
- `def-stance-mgmt-hold-then-attack` — Mid hold-fire→return-fire timing // ROE drill — YAML
- `def-while-building` — Mid defend with workforce while building defenses // ops continuity — YAML
- `def-multi-direction` — Mid defend 3 lanes // distributed defense; graph min-cut — YAML
- `def-in-depth` — Mid-Late front + 2nd line vs single thick // defense-in-depth; security doctrine — YAML
- `def-with-ambush` — Mid fog ambush vs static // active deception; SC2 hidden — YAML
- `def-evacuation` — Mid-Late evacuate overwhelmed position // BCP; emergency mgmt — YAML
- `def-retreat-and-rebuild` — Mid-Late concede + rebuild deeper // strategic withdrawal; CICERO retreat — YAML
- `def-anti-air-positioning` — Late SAM placement // anti-air; SC2 — ENG:air-units + YAML
- `def-tower-rush-counter` — Opening enemy tower-rushes you // tower-rush defense; SC2 cannon-rush counter — BOT:tower-rusher + YAML

## Group D — Combat micro & tactics (20)

- `combat-focus-fire-priority` — Mid which enemy to kill first // target prioritization; SC2 focus-fire — YAML
- `combat-kite-jeep-vs-tank` — Mid kite ranged // micro; SC2 kiting — YAML
- `combat-stance-mgmt-attack` — Mid switch to attack-anything // ROE; SC2 stance — YAML
- `combat-formation-tank-wedge` — Mid tank wedge // formation; military — YAML
- `combat-formation-infantry-screen-tanks` — Mid infantry screens armor // combined-arms; military — YAML
- `combat-retreat-after-engagement` — Mid disengage timing // tactical withdrawal; SC2 — YAML
- `combat-flanking-attack` — Mid flank vs head-on // flank maneuver; military — YAML
- `combat-pincer-coordination` — Mid two-prong // pincer; SC2 multi-prong — YAML
- `combat-bait-counter-attack` — Mid lure + flank-counter // bait; SC2 bait — BOT:guard + YAML (idiom proven)
- `combat-divide-and-conquer` — Mid split enemy via feint // CICERO — YAML
- `combat-protect-vip-escort` — Mid escort VIP // diplomatic escort — PRED:`unit_type_count_gte:{vip,1}`-survival + YAML
- `combat-suicide-charge-mission` — Mid sacrifice everything for X // forlorn hope; military — YAML
- `combat-prevent-retreat` — Mid cut off enemy retreat // encirclement; military — YAML
- `combat-defensive-perimeter-around-vip` — Mid perimeter // VIP cordon — YAML
- `combat-skirmish-then-disengage` — Mid bleed and pull back // skirmisher; SC2 — YAML
- `combat-vehicle-vs-infantry-counter` — Mid tank vs inf // RPS counter; SC2 hard counter — YAML
- `combat-rocket-soldier-anti-vehicle` — Mid e3 vs tank // RPS; SC2 — YAML
- `combat-tank-vs-tank-engagement` — Mid mirror combat // SC2 mirror micro — YAML
- `combat-amphibious-landing` — Mid-Late APC across water // amphibious assault; D-Day — ENG:transport+water-mapgen + YAML
- `combat-attack-from-behind-fog` — Mid emerge from fog // surprise attack; SC2 hidden — YAML

## Group E — Scouting / Perception (15)

- `scout-far-frontier` — Early long-distance scout // recon; ERQA — YAML
- `scout-and-survive` — Early scout returns alive // scout-preserve; SC2 worker scout — YAML
- `scout-multiple-fog-areas` — Early parallel scout // Watch-And-Help — YAML
- `scout-detect-enemy-tech` — Early-Mid identify enemy via buildings // tech-read; PlanBench obs — YAML
- `scout-detect-base-direction` — Early which corner has enemy // direction-read; ERQA — YAML
- `scout-count-defenders` — Early-Mid exact defender count // census; POMDP — YAML
- `scout-discover-hidden-base` — Mid off-axis hidden base // deep recon; CICERO — YAML
- `scout-detect-incoming-army` — Mid early-warn // intrusion detection — YAML
- `scout-detect-air-threat` — Late air detection // anti-air recon — ENG:air-units + YAML
- `scout-detect-stealth-units` — Late stealth detection // SC2 detector — ENG:stealth-units + YAML
- `scout-map-reveal-percent-target` — Early reveal N% // coverage; ERQA — YAML
- `scout-and-report` — Early-Mid reveal then return // recon-report cycle; mil intel — YAML
- `scout-jeep-vs-infantry-cost-effective` — Early cheap fast scout // econ scout; SC2 worker-vs-Reaper — YAML
- `scout-cycle-keep-info-fresh` — Mid continuous cadence // info staleness; intel ops — PRED:then-repeated + YAML
- `scout-info-blackout-react` — Mid fog blanket then enemy revealed // dynamic perception; adversarial — ENG:fog-shift-hook + YAML

## Group F — Economy / Resources (16)

Mostly unblocked by S0/S1 (task #14).

- `econ-startup-from-scratch` — Opening first proc + harv // greenfield econ; SC2 first base — YAML
- `econ-harvester-pathing-optimization` — Opening harv route choice // logistics; queueing — PRED:`econ_rate_gte` + YAML
- `econ-multi-patch-allocation` — Mid which ore patches // resource allocation; OR — YAML
- `econ-far-patch-vs-near-patch` — Mid travel vs throughput // econ tradeoff; OR — YAML
- `econ-silo-vs-spend` — Mid store or spend // capital allocation; PlanBench — YAML
- `econ-replace-dead-harvester` — Mid rebuild dead harv // resilience; PlanBench replanning — YAML
- `econ-protect-harvester-route` — Mid defend harv path // econ defense; SC2 harass defense — YAML
- `econ-cash-reserve-management` — Mid never zero cash // financial reserve — PRED:`cash_gte` + YAML
- `econ-target-cash-amount-by-deadline` — Mid cash bar by deadline // budget by date; finance — YAML
- `econ-contention-with-enemy` — Mid shared ore patch // contested resource; game theory — YAML
- `econ-recover-from-zero-cash` — Mid start from 0 cash // bankruptcy comeback — YAML
- `econ-quantitative-vs-qualitative-spend` — Mid many cheap vs few expensive // unit composition; OR — YAML
- `econ-burn-rate-management` — Mid spend at target rate // runway mgmt; startup finance — PRED:`spend_rate_lte` + YAML
- `econ-buy-vs-build-decision` — Mid cash now or save for tech // PlanBench — YAML
- `econ-resource-trade-with-self` — Mid-Late balance ore vs cash // commodity hedging; finance — YAML
- `econ-overflow-to-silos` — Mid silo-required throughput // capacity planning; backpressure — YAML

## Group G — Long-horizon / Multi-phase (12)

- `lh-opening-to-tech-to-army` — Late 4-phase chain // long-horizon; SC2 macro — YAML
- `lh-opening-to-defense-to-counter` — Late defense → counter // counter-attack timing; SC2 timing — YAML
- `lh-tech-pivot-attack` — Late tech → pivot → attack // adaptive macro; PlanBench — YAML
- `lh-econ-army-victory` — Late econ → army → victory // SC2 macro — YAML
- `lh-defense-tech-second-base` — Late defend → tech → expand // SC2 secure-expand — YAML
- `lh-scout-react-counter` — Late scout → react → counter // CICERO — YAML
- `lh-100-turn-marathon-survival` — Very-Late survive 100 turns // lmgame-Bench multi-hour — YAML (max_turns:100+)
- `lh-build-army-coordinate-multifront-attack` — Late assemble + coordinate // operational planning; military — YAML
- `lh-progression-stage-locked` — Late each stage gates next // PERT; PlanBench — PRED:then + YAML
- `lh-credit-only-final-phase` — Late sparse reward // long-horizon RL — YAML
- `lh-multi-checkpoint-5-plus` — Late 5+ checkpoints // sequenced ops; PlanBench — PRED:waypoint_sequence-extended + YAML
- `lh-recovery-after-mid-game-loss` — Late comeback // SC2 comeback — ENG:scheduled-event + YAML

## Group H — Adversarial / Game-theoretic (12)

- `adv-counter-strategy-read` (= C1 from plan)
- `adv-feint-handling` (= C4 from plan)
- `adv-bluff-detection` — Mid commit vs hesitate // poker-bench — ENG:RL-adversary + YAML
- `adv-rps-counter-pick` — Mid unit-counter selection // RPS; game theory — YAML
- `adv-tempo-vs-adversary` — Mid-Late out-tempo // TextStarCraft II — ENG:RL-adversary + YAML
- `adv-economy-race` — Late out-econ // SC2 econ race — ENG:RL-adversary + YAML
- `adv-army-race` — Late out-army // SC2 army race — ENG:RL-adversary + YAML
- `adv-tech-race` — Late out-tech // SC2 tech race — ENG:RL-adversary + YAML
- `adv-base-trade-race` (= C3 from plan)
- `adv-asymmetric-weaker-must-win` — Mid outnumbered must-win // underdog; RTS asymmetric — YAML
- `adv-asymmetric-stronger-no-loss` — Mid dominate without loss // overdog discipline; CICERO — YAML
- `adv-mirror-match-best-strategy` — Late mirror best-of // SC2 mirror — ENG:RL-adversary + YAML

## Group I — Procedural / Tool-use compliance (10)

- `proc-strict-toolban-fidelity` — Cross BFCL relevance // BFCL V4 — PRED:forbidden_tools + YAML
- `proc-strict-toolban-under-pressure` (= B3 from plan)
- `proc-checklist-no-deviation` — Cross exact ordered checklist // SOP compliance; IFBench — PRED:waypoint_sequence strict + YAML
- `proc-ordered-action-strict` — Cross exact action sequence // PlanBench strict ordering — PRED:then + YAML
- `proc-no-attack-passive-only` — Cross only move+stop // ROE; mil — PRED:forbidden_tools + YAML
- `proc-only-build-no-combat` — Cross only build/place // role-based access — PRED:forbidden_tools + YAML
- `proc-only-defend-no-attack` — Cross only defensive moves // ROE; mil — PRED:forbidden_tools + YAML
- `proc-tool-use-with-distractor` — Cross ignore distracting tool // τ²-bench — YAML
- `proc-tool-use-multi-distractor` — Cross ignore multiple distractors // τ²-bench — YAML
- `proc-instruction-following-edge-case` — Cross instruction-edge // IFBench — YAML

## Group J — Robustness / Edge cases (8)

- `rob-unit-loss-recovery` — Mid recover from N losses // resilience; PlanBench replanning — YAML
- `rob-building-loss-rebuild` (= C5 from plan)
- `rob-cash-depletion-recovery` — Mid recover from 0 cash // bankruptcy — YAML
- `rob-power-failure-recovery` — Mid recover from brownout // grid restoration; DR — ENG:power-signals + YAML
- `rob-multiple-simultaneous-pressures` — Mid econ + def + tech under pressure // overwhelmed ops; triage — YAML
- `rob-unexpected-enemy-spawn` — Mid spawn mid-episode // surprise; adversarial robustness — ENG:scheduled-spawn + YAML
- `rob-objective-change-midway` — Mid win-condition shifts at T // goal-conditional RL — ENG:objective-swap-hook + YAML
- `rob-deadline-shortened-midway` — Mid deadline brought forward // schedule compression; project mgmt — ENG:dynamic-deadline-hook + YAML

## Group K — Special abilities (engine-gated, 10)

- `spec-engineer-capture` — Mid capture enemy `proc` // industrial espionage; SC2 spy steal — ENG:capture_actor + YAML
- `spec-engineer-mass-capture` — Mid capture all of enemy tech // takeover; CICERO — ENG:capture_actor + YAML
- `spec-spy-infiltrate` — Mid reveal enemy tech via spy // intel infiltration; SC2 scan — ENG:spy + YAML
- `spec-thief-steal-resources` — Mid steal enemy cash // econ sabotage; SC2 worker raid — ENG:thief + YAML
- `spec-medic-heal-coordination` — Mid medic heals infantry // SC2 medivac — ENG:medic + YAML
- `spec-airborne-paradrop` — Late drop reinforcements // SC2 paradrop — ENG:airborne + YAML
- `spec-nuke-strike` — Very-late superweapon // SC2 nuke — ENG:superweapon + YAML
- `spec-iron-curtain-buff` — Late invuln buff // SC2 cloak — ENG:iron-curtain + YAML
- `spec-chronosphere-teleport` — Late unit teleport // SC2 warpgate — ENG:chronosphere + YAML
- `spec-spy-detect-tech` — Mid spy + tech reveal // counterintel; CICERO — ENG:spy + YAML

## Group L — Transport / Amphibious (6)

- `transport-apc-deliver-infantry` — Mid APC delivers infantry // last-mile delivery; logistics — ENG:apc-transport + YAML
- `transport-amphibious-landing` — Mid-Late LST water cross // amphibious; D-Day — ENG:water-mapgen+lst + YAML
- `transport-evacuate-by-transport` — Mid evac units // BCP evacuation; emergency — ENG:apc-transport + YAML
- `transport-multiple-trips` — Mid ferry multiple loads // queueing — ENG:apc-transport + YAML
- `transport-protect-en-route` — Mid escort transport // convoy escort; mil — ENG:apc-transport + YAML
- `transport-amphibious-island-base` — Mid-Late build on island // amphibious base; strategic geography — ENG:water-mapgen+lst + YAML

## Group M — Power / Maintenance (7)

Most are gated on the engine power-signals fix (`power_surplus_gte`
is currently inert — see CLAUDE.md footguns).

- `power-budget-online` — Opening keep `power_surplus_gte:0` // grid bring-up; SC2 power — ENG:power-signals + YAML
- `power-down-non-essential-under-attack` — Mid load-shed // ops runbook — ENG:power-signals + YAML
- `power-restoration-order-after-loss` — Mid restore in correct order // utility restoration; DR — ENG:power-signals + YAML
- `power-tech-vs-defense-conflict` — Mid limited power: tech vs def // capacity allocation; OR — ENG:power-signals + YAML
- `maint-repair-priority-order` — Mid repair triage // DR — YAML
- `maint-sell-and-recoup-cash` — Mid sell + recoup // capital reallocation; finance — YAML
- `maint-rebuild-from-power-failure` — Mid restore after brownout // DR; utility — ENG:power-signals + YAML

## Group N — Coordination & Multi-agent (8)

- `coord-squad-handoff` — Mid squad A to squad B handoff // multi-agent handoff; Watch-And-Help — YAML
- `coord-relay-attack` — Mid A attacks, B follows up // relay; SMAC — YAML
- `coord-mutual-support` — Mid stay in mutual support range // mil doctrine — PRED:max_inter_unit_distance_lte + YAML
- `coord-converge-on-target` — Mid 3 squads converge // SMAC — YAML
- `coord-cover-and-move` — Mid A covers while B moves // fire-and-maneuver; mil — YAML
- `coord-diversionary-attack` — Mid A diverts, B attacks real target // CICERO — YAML
- `coord-air-ground-combined` — Late (air) combined-arms // mil — ENG:air-units + YAML
- `coord-naval-ground-combined` — Late (naval) combined naval-ground // D-Day — ENG:naval + YAML

## Group O — Time-pressure variants (8)

(Each is a "+ tight clock" variant of an existing scenario family.)

- `tp-rush-objective-very-fast` — Mid speedrun objective // speedrun; human speedrun — YAML
- `tp-rush-objective-tighter` — Mid tighter speedrun — YAML
- `tp-rush-multi-objective` — Mid speedrun 2 objectives // speedrun parallel — YAML
- `tp-survive-N-turns` — Mid survive until clock // survival; SC2 survival — YAML
- `tp-survive-and-grow` — Mid survive + grow // ops continuity; SRE — YAML
- `tp-survive-and-strike-at-window` — Mid survive then strike // SC2 timing window — PRED:then + YAML
- `tp-decision-under-clock` — Mid quick correct choice // tense decision; poker — YAML
- `tp-pressure-procedural` — Cross procedural under tight clock // procedure under stress; IFBench — PRED:forbidden_tools + YAML
