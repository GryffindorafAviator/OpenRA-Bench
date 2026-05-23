# Scenario Categories — Skills × Tools

Bucketing of all **210 active scenario packs** in
`openra_bench/scenarios/packs/` (excluding the 3 archived packs in
`_archive/`) by inferred **skill family** and **tool surface**.

The intended use is the human audit pass: walk one category at a time,
checking each pack's objective, win / lose predicate, custom-map
presence, and level matrix against the tools the YAML actually exposes
to the model.

Generated 2026-05-23 from a YAML walk that:

* derives the **family** from the filename prefix (`combat-`, `def-`,
  `econ-`, ...);
* collapses near-synonym families (`coord-` and `coordination-` →
  Coordination, `lh-` and `longhorizon-` → Long-horizon, etc.) into 12
  named categories;
* collects the **union of every `tools:` list** in each pack — base,
  per-level, and overrides — so a tool exposed only on `hard` still
  shows up.

The full audit script is reproducible:

```bash
ls openra_bench/scenarios/packs/*.yaml \
  | xargs -n1 basename | sed 's/\.yaml$//' \
  | awk -F'-' '{print $1}' | sort | uniq -c | sort -rn
```

…and the per-pack table at the bottom is what `scripts/audit_categories.py` would produce
(the one-shot Python the audit was generated with is inlined in the
report attached to this file's first commit).


## 12 categories (210 packs)

| # | Category | Packs | Required tool groups | Tool union |
|---|---|---:|---|---|
| 1 | **Economy** | 38 | movement, combat, build, mcv, economy, observe | `attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, sell, set_rally_point, set_stance, stop` |
| 2 | **Build / Construction** | 32 | movement, combat, build, mcv, economy, structure, observe | `attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, power_down, repair, sell, set_primary, set_rally_point, stop` |
| 3 | **Defense** | 29 | movement, combat, build, mcv, economy, structure, observe | `attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, repair, sell, set_stance, stop, stop_units` |
| 4 | **Combat** | 25 | movement, combat, build, observe | `attack_move, attack_unit, build, move_units, observe, place_building, set_stance, stop` |
| 5 | **Scouting** | 19 | movement, combat, build, observe | `attack_move, attack_unit, build, move_units, observe, place_building, set_stance, stop, stop_units` |
| 6 | **Long-horizon** | 14 | movement, combat, build, mcv, economy, structure, observe | `attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, repair, sell, set_stance, stop, stop_units` |
| 7 | **Tempo / Timing** | 13 | movement, combat, build, economy, observe | `attack_move, attack_unit, build, cancel_production, harvest, move_units, observe, place_building, set_stance, stop, stop_units, train` |
| 8 | **Coordination** | 11 | movement, combat, observe | `attack_move, attack_unit, move_units, observe, stop, stop_units` |
| 9 | **Tech / Decision** | 11 | movement, combat, build, mcv, economy, observe | `attack_move, attack_unit, build, cancel_production, deploy, harvest, move_units, observe, place_building, set_rally_point, stop, stop_units, train` |
| 10 | **Adversarial** | 10 | movement, combat, build, mcv, observe | `attack_move, attack_unit, build, deploy, move_units, observe, place_building, stop` |
| 11 | **Special / Superweapon** | 6 | movement, combat, build, observe, special | `attack_unit, build, c4_detonate, capture_actor, fire_superweapon, infiltrate, move_units, observe, place_building, set_stance, stop` |
| 12 | **Misc** | 2 | movement, combat | `attack_unit, move_units, stop, stop_units` |

`set_stance` and `stop_units` appear next to `stop` because some YAML
packs use the alias name; the agent translation layer treats them as
equivalent.


## Family-prefix → category map

| Family prefix | Count | Category |
|---|---:|---|
| `combat-` | 25 | Combat |
| `def-`, `defense-`, `rob-`, `maint-` | 18+1+8+2 = 29 | Defense |
| `scout-`, `perception-`, `navigation-` | 14+4+1 = 19 | Scouting |
| `build-`, `building-`, `expansion-`, `mfb-`, `mcv-` | 14+1+3+8+6 = 32 | Build / Construction |
| `econ-`, `economy-`, `proc-` | 23+5+10 = 38 | Economy |
| `lh-`, `longhorizon-` | 13+1 = 14 | Long-horizon |
| `tp-`, `tempo-`, `rush-`, `mid-` | 7+2+1+3 = 13 | Tempo / Timing |
| `tech-`, `reasoning-`, `strategy-`, `risk-` | 4+2+4+1 = 11 | Tech / Decision |
| `coord-`, `coordination-`, `action-` | 7+2+2 = 11 | Coordination |
| `spec-`, `power-` | 5+1 = 6 | Special / Superweapon |
| `adv-`, `adversarial-`, `artofwar-`, `strict-`, `harass-` | 2+1+3+3+1 = 10 | Adversarial |
| `custom-`, `TEMPLATE` | 1+1 = 2 | Misc |


## Tool-skill rosetta

| Tool group | Tools | What it tests |
|---|---|---|
| **movement** | `move_units`, `attack_move`, `patrol`, `stop`, `stop_units` | path-finding, formation, retreat |
| **combat** | `attack_unit`, `set_stance` | target priority, focus-fire, hold-vs-attack stance |
| **build** | `build`, `place_building`, `cancel_production`, `set_rally_point`, `set_primary` | production sequencing, building placement, rally management |
| **mcv** | `deploy` | bootstrap from MCV → `fact` |
| **economy** | `harvest`, `sell` | resource routing, sell-and-recoup |
| **structure** | `repair`, `power_down` | base-state mgmt under fire |
| **special** | `fire_superweapon`, `infiltrate`, `c4_detonate`, `capture_actor`, `unload`, `surrender` | superweapons, hero units, unit-on-unit specials |
| **observe** | `observe` | safe no-op — always available so the model can pass a turn |


## Cross-bucket flags worth a closer look during the audit

* **`adversarial-duel`** declares an empty tool list — the YAML
  probably defers tool selection to an inner sub-pack. Confirm during
  the Adversarial walk; this could be a real authoring gap.
* **`combat-harass-balanced-hit-and-run`** is the only Combat pack
  missing `attack_move`. Kite-only with explicit `attack_unit` + `stop`
  — intentional? Check on the Combat walk.
* **`mfb-base-1-defend-base-2-build`** has "build" in the name but no
  `build` / `place_building` in its tool list — name says one thing,
  allowlist says another. Audit during Build / Construction.
* **`maint-repair-priority-order`** is the only Defense pack with just
  `[observe, repair]`. Confirm the win predicate is solvable under that
  allowlist (no offensive verbs at all).
* **`adv-asymmetric-weaker-must-win`** and **`harass-response-preserve`**
  are the only Adversarial packs without `build` / `place_building`.
  Pure tactical, no construction — verify intent.


## Per-category pack lists

### Economy (38)

`union: attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, sell, set_rally_point, set_stance, stop`

| Pack | Tools |
|---|---|
| `econ-burn-rate-management` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `econ-buy-vs-build-decision` | `attack_move,attack_unit,build,move_units,place_building,stop` |
| `econ-cash-reserve-management` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-contention-with-enemy` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `econ-contested-expansion` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,sell,set_stance,stop` |
| `econ-deny-enemy-expansion` | `attack_move,attack_unit,move_units,stop` |
| `econ-expansion-timing` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-far-patch-vs-near-patch` | `harvest,move_units,observe,stop` |
| `econ-harvester-defense-raid` | `attack_move,attack_unit,harvest,move_units,observe,set_stance,stop` |
| `econ-harvester-pathing-optimization` | `harvest,move_units,observe,stop` |
| `econ-mine-and-grow` | `build,move_units,observe,place_building,stop` |
| `econ-multi-patch-allocation` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-overflow-to-silos` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-protect-harvester-route` | `attack_move,attack_unit,harvest,move_units,observe,stop` |
| `econ-quantitative-vs-qualitative-spend` | `attack_move,attack_unit,build,move_units,place_building,stop` |
| `econ-recover-from-zero-cash` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-replace-dead-harvester` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-resource-trade-with-self` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-second-base-race` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-silo-vs-spend` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `econ-startup-from-scratch` | `build,deploy,harvest,move_units,observe,place_building,stop` |
| `econ-target-cash-amount-by-deadline` | `build,harvest,move_units,observe,place_building,stop` |
| `econ-tech-vs-expand-decision` | `build,harvest,move_units,observe,place_building,stop` |
| `economy-force-buildup` | `build,deploy,move_units,place_building,stop` |
| `economy-harvest-investment` | `build,deploy,harvest,move_units,place_building,stop` |
| `economy-harvest-timebox` | `build,harvest,move_units,place_building,stop` |
| `economy-investment` | `build,deploy,move_units,place_building,stop` |
| `economy-time-box` | `build,deploy,move_units,place_building,stop` |
| `proc-checklist-no-deviation` | `move_units,observe,stop` |
| `proc-conditional-branch-action` | `attack_move,attack_unit,move_units,observe,stop` |
| `proc-instruction-following-edge-case` | `attack_unit,move_units,observe,stop` |
| `proc-no-attack-passive-only` | `move_units,observe,stop` |
| `proc-only-build-no-combat` | `build,move_units,observe,place_building,stop` |
| `proc-only-defend-no-attack` | `move_units,observe,set_stance,stop` |
| `proc-ordered-action-strict` | `build,observe,place_building` |
| `proc-strict-toolban-fidelity` | `build,move_units,observe,place_building,stop` |
| `proc-tool-use-multi-distractor` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,set_rally_point,set_stance` |
| `proc-tool-use-with-distractor` | `attack_unit,build,move_units,observe,place_building` |


### Build / Construction (32)

`union: attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, power_down, repair, sell, set_primary, set_rally_point, stop`

| Pack | Tools |
|---|---|
| `build-defensive-skirt-corners` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `build-defensive-tower-cluster` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `build-defensive-tower-line` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `build-engineer-rebuild-after-loss` | `attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `build-power-down-defensive` | `observe,power_down,sell,stop` |
| `build-power-online-first` | `build,observe,place_building,stop` |
| `build-production-throughput-multibuilding` | `build,observe,place_building,stop` |
| `build-rally-point-management` | `build,observe,set_rally_point,stop` |
| `build-repair-priority-under-fire` | `observe,repair` |
| `build-sell-and-rebuild-elsewhere` | `build,observe,place_building,sell` |
| `build-sequence-tech-cheapest` | `build,observe,place_building` |
| `build-sequence-tech-fastest` | `build,observe,place_building` |
| `build-sequence-tech-most-resilient` | `build,harvest,move_units,observe,place_building,stop` |
| `build-tech-skip-decision` | `attack_move,build,move_units,observe,place_building,stop` |
| `building-and-planning` | `build,move_units,place_building,stop` |
| `expansion-aggro-3-base-greedy` | `build,deploy,harvest,move_units,observe,place_building,stop` |
| `expansion-balanced-2-base-defended` | `attack_unit,build,deploy,move_units,observe,place_building,stop` |
| `expansion-turtle-1-base-fortified` | `attack_unit,build,move_units,observe,place_building,stop` |
| `mcv-deploy-and-build` | `build,deploy,move_units,observe,place_building,stop` |
| `mcv-deploy-defensible-site` | `build,deploy,move_units,observe,place_building,stop` |
| `mcv-deploy-near-resource` | `build,deploy,harvest,move_units,observe,place_building,stop` |
| `mcv-deploy-relocate-under-pressure` | `deploy,move_units,observe` |
| `mcv-deploy-second-base` | `build,deploy,move_units,observe,place_building,stop` |
| `mcv-deploy-third-base` | `build,deploy,move_units,observe,place_building,stop` |
| `mfb-base-1-defend-base-2-build` ⚠️ | `attack_move,attack_unit,deploy,move_units,observe,stop` |
| `mfb-mirror-base-east-west` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `mfb-redundant-tech-buildings` | `build,harvest,move_units,observe,place_building,stop` |
| `mfb-rotating-production-pressure` | `build,observe,set_primary,set_rally_point` |
| `mfb-supply-line-link-between-bases` | `attack_move,attack_unit,move_units,observe,stop` |
| `mfb-tech-base-vs-economy-base` | `build,harvest,move_units,observe,place_building,stop` |
| `mfb-third-base-against-clock` | `build,harvest,move_units,observe,place_building,stop` |
| `mfb-two-base-simultaneous` | `build,observe,place_building,stop` |

⚠️ `mfb-base-1-defend-base-2-build` — name implies build, allowlist
omits `build`/`place_building`. Probable authoring gap.


### Defense (29)

`union: attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, repair, sell, set_stance, stop, stop_units`

| Pack | Tools |
|---|---|
| `def-bridge-chokepoint` | `attack_move,attack_unit,move_units,observe,set_stance,stop` |
| `def-counter-battery` | `attack_move,attack_unit,move_units,observe,stop` |
| `def-engineer-repair-under-fire` | `attack_move,attack_unit,build,move_units,observe,place_building,repair,set_stance,stop` |
| `def-evacuation` | `attack_move,attack_unit,move_units,observe,stop` |
| `def-in-depth-vs-single` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-in-depth` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-multi-direction` | `attack_move,attack_unit,move_units,observe,stop` |
| `def-position-expected-direction` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-position-revealed-direction` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-pre-position-mobile-reserve` | `attack_move,attack_unit,move_units,observe,stop` |
| `def-reinforce-the-breach` | `attack_move,attack_unit,move_units,observe,stop` |
| `def-retreat-and-rebuild` | `attack_move,attack_unit,build,deploy,move_units,observe,place_building,stop` |
| `def-stance-mgmt-hold-then-attack` | `attack_move,attack_unit,move_units,observe,set_stance,stop` |
| `def-surprise-flank-react` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-tower-line-vs-cluster` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-walls-vs-towers` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-while-building` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `def-with-ambush` | `attack_move,attack_unit,move_units,observe,set_stance,stop` |
| `defense-rush-survive` | `attack_move,attack_unit,build,move_units,place_building,stop` |
| `maint-repair-priority-order` ⚠️ | `observe,repair` |
| `maint-sell-and-recoup-cash` | `build,observe,place_building,sell` |
| `rob-cash-depletion-recovery` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `rob-deadline-shortened-midway` | `attack_move,attack_unit,move_units,observe,stop` |
| `rob-multiple-simultaneous-pressures` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `rob-objective-change-midway` | `attack_move,attack_unit,build,move_units,observe,place_building,stop_units` |
| `rob-objective-shift-with-or-clause` | `attack_move,attack_unit,build,move_units,observe,place_building,stop_units` |
| `rob-partial-base-loss-continue` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `rob-unexpected-enemy-spawn` | `attack_move,attack_unit,move_units,observe,stop` |
| `rob-unit-loss-recovery` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |

⚠️ `maint-repair-priority-order` — only `[observe, repair]`. Confirm the
win predicate is reachable.


### Combat (25)

`union: attack_move, attack_unit, build, move_units, observe, place_building, set_stance, stop`

| Pack | Tools |
|---|---|
| `combat-attack-from-behind-fog` | `attack_move,attack_unit,move_units,stop` |
| `combat-bait-counter-attack` | `attack_move,attack_unit,move_units,stop` |
| `combat-divide-and-conquer` | `attack_move,attack_unit,move_units,stop` |
| `combat-flanking-attack` | `attack_move,attack_unit,move_units,stop` |
| `combat-focus-fire-priority` | `attack_move,attack_unit,move_units,stop` |
| `combat-formation-tank-wedge` | `attack_move,attack_unit,move_units,stop` |
| `combat-harass-aggro-commit` | `attack_move,attack_unit,move_units,stop` |
| `combat-harass-balanced-hit-and-run` ⚠️ | `attack_unit,move_units,stop` |
| `combat-heli-flank` | `attack_move,attack_unit,move_units,stop` |
| `combat-hold-chokepoint` | `attack_move,attack_unit,move_units,observe,stop` |
| `combat-kite-and-pull` | `attack_move,attack_unit,move_units,observe,stop` |
| `combat-kite-jeep-vs-tank` | `attack_move,attack_unit,move_units,stop` |
| `combat-naval-shore-strike` | `attack_move,attack_unit,move_units,stop` |
| `combat-pincer-coordination` | `attack_move,attack_unit,move_units,stop` |
| `combat-prevent-retreat` | `attack_move,attack_unit,move_units,stop` |
| `combat-protect-vip-escort` | `attack_move,attack_unit,move_units,stop` |
| `combat-retreat-after-engagement` | `attack_move,attack_unit,move_units,stop` |
| `combat-rocket-soldier-anti-vehicle` | `attack_move,attack_unit,build,move_units,place_building,stop` |
| `combat-skirmish-then-disengage` | `attack_move,attack_unit,move_units,stop` |
| `combat-stance-mgmt-attack` | `attack_move,attack_unit,move_units,observe,set_stance,stop` |
| `combat-suicide-charge-mission` | `attack_move,attack_unit,move_units,stop` |
| `combat-tank-vs-tank-engagement` | `attack_move,attack_unit,move_units,stop` |
| `combat-tanya-vs-rush` | `attack_move,attack_unit,move_units,set_stance,stop` |
| `combat-target-priority-highvalue` | `attack_move,attack_unit,move_units,stop` |
| `combat-vehicle-vs-infantry-counter` | `attack_move,attack_unit,build,move_units,place_building,stop` |

⚠️ `combat-harass-balanced-hit-and-run` is the only Combat pack missing
`attack_move`. Intentional kite-only? Confirm.


### Scouting (19)

`union: attack_move, attack_unit, build, move_units, observe, place_building, set_stance, stop, stop_units`

| Pack | Tools |
|---|---|
| `navigation-confined-hard-only` | `move_units,stop` |
| `perception-count-the-threat-small-k` | `attack_unit,move_units,stop_units` |
| `perception-count-the-threat` | `build,move_units,observe,place_building,set_stance,stop` |
| `perception-frontier-reading` | `attack_unit,move_units,stop_units` |
| `perception-target-vs-fog` | `attack_unit,move_units,stop_units` |
| `scout-and-report` | `attack_unit,move_units,stop` |
| `scout-and-survive` | `move_units,stop` |
| `scout-count-defenders` | `attack_move,attack_unit,build,move_units,observe,place_building,stop_units` |
| `scout-cycle-keep-info-fresh` | `attack_move,attack_unit,move_units,observe,stop` |
| `scout-deny-enemy-vision` | `attack_move,attack_unit,move_units,observe,stop` |
| `scout-detect-base-direction` | `attack_move,attack_unit,move_units,observe,stop` |
| `scout-detect-enemy-tech` | `attack_unit,move_units,stop` |
| `scout-detect-incoming-army` | `attack_move,attack_unit,move_units,observe,stop` |
| `scout-discover-hidden-base` | `attack_unit,move_units,stop` |
| `scout-far-frontier` | `move_units,stop` |
| `scout-jeep-vs-infantry-cost-effective` | `build,move_units,stop` |
| `scout-map-reveal-percent-target` | `move_units,observe,stop` |
| `scout-multiple-fog-areas` | `attack_unit,move_units,stop` |
| `scout-track-enemy-movement` | `attack_move,attack_unit,move_units,observe,stop` |


### Long-horizon (14)

`union: attack_move, attack_unit, build, deploy, harvest, move_units, observe, place_building, repair, sell, set_stance, stop, stop_units`

| Pack | Tools |
|---|---|
| `lh-100-turn-marathon-survival` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,repair,sell,stop` |
| `lh-build-army-coordinate-multifront-attack` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `lh-credit-only-final-phase` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `lh-defense-tech-second-base` | `attack_move,attack_unit,build,deploy,harvest,move_units,observe,place_building,stop` |
| `lh-econ-army-victory` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,set_stance,stop` |
| `lh-multi-checkpoint-5-plus` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `lh-opening-to-defense-to-counter` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,set_stance,stop` |
| `lh-opening-to-tech-to-army` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `lh-progression-stage-locked` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,set_stance,stop` |
| `lh-recovery-after-mid-game-loss` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `lh-scout-react-counter` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `lh-tech-pivot-attack` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |
| `lh-tech-rush-vs-army-rush` | `attack_move,attack_unit,build,move_units,observe,place_building,stop` |
| `longhorizon-opening-to-assault` | `attack_unit,build,deploy,move_units,place_building,stop_units` |


### Tempo / Timing (13)

`union: attack_move, attack_unit, build, cancel_production, harvest, move_units, observe, place_building, set_stance, stop, stop_units, train`

| Pack | Tools |
|---|---|
| `mid-concede-vs-hold` | `attack_move,attack_unit,move_units,stop_units` |
| `mid-economy-under-fire` | `attack_move,attack_unit,harvest,move_units,observe,stop` |
| `mid-tech-switch-on-scout` | `attack_move,attack_unit,build,cancel_production,move_units,place_building,stop_units,train` |
| `rush-hour` | `attack_move,attack_unit,move_units,stop_units` |
| `tempo-double-window` | `attack_move,attack_unit,move_units,stop` |
| `tempo-strike-window` | `attack_unit,move_units,stop_units` |
| `tp-decision-under-clock` | `attack_move,attack_unit,move_units,stop_units` |
| `tp-pressure-procedural` | `move_units,observe,set_stance,stop` |
| `tp-rush-multi-objective` | `attack_move,attack_unit,move_units,stop_units` |
| `tp-rush-objective-very-fast` | `attack_move,attack_unit,move_units,stop_units` |
| `tp-survive-and-grow` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,set_stance,stop` |
| `tp-survive-and-strike-at-window` | `attack_move,attack_unit,move_units,stop_units` |
| `tp-survive-n-turns` | `attack_move,attack_unit,move_units,stop` |


### Coordination (11)

`union: attack_move, attack_unit, move_units, observe, stop, stop_units`

| Pack | Tools |
|---|---|
| `action-multiunit-coordination` | `attack_unit,move_units,stop` |
| `action-sequenced-execution` | `attack_unit,move_units,stop` |
| `coord-converge-on-target` | `attack_move,attack_unit,move_units,stop` |
| `coord-cover-and-move` | `attack_move,attack_unit,move_units,stop` |
| `coord-diversionary-attack` | `attack_move,attack_unit,move_units,stop` |
| `coord-mutual-support` | `attack_move,attack_unit,move_units,stop` |
| `coord-relay-attack` | `attack_move,attack_unit,move_units,stop` |
| `coord-relay-vision-chain` | `move_units,observe,stop` |
| `coord-squad-handoff` | `attack_move,attack_unit,move_units,stop` |
| `coordination-ordered-rendezvous` | `attack_unit,move_units,stop_units` |
| `coordination-staggered-window` | `attack_unit,move_units,stop` |


### Tech / Decision (11)

`union: attack_move, attack_unit, build, cancel_production, deploy, harvest, move_units, observe, place_building, set_rally_point, stop, stop_units, train`

| Pack | Tools |
|---|---|
| `reasoning-frontier-commit` | `attack_unit,move_units,stop_units` |
| `reasoning-risk-route` | `attack_unit,move_units,stop_units` |
| `risk-blockade-bypass` | `attack_unit,move_units,stop_units` |
| `strategy-dilemma` | `attack_unit,move_units,stop_units` |
| `strategy-gauntlet` | `attack_move,attack_unit,move_units,stop_units` |
| `strategy-trilemma` | `build,move_units,place_building,stop` |
| `strategy-twobody` | `attack_move,attack_unit,move_units,stop_units` |
| `tech-aggro-all-in` | `build,move_units,observe,place_building,stop` |
| `tech-balanced-econ-then-tech` | `build,harvest,move_units,observe,place_building,stop` |
| `tech-production-planning` | `build,cancel_production,deploy,move_units,place_building,set_rally_point,stop,train` |
| `tech-turtle-defensive-tech` | `attack_move,attack_unit,build,harvest,move_units,observe,place_building,stop` |


### Adversarial (10)

`union: attack_move, attack_unit, build, deploy, move_units, observe, place_building, stop`

| Pack | Tools |
|---|---|
| `adv-asymmetric-weaker-must-win` ⚠️ | `attack_move,attack_unit,move_units,stop` |
| `adv-rps-counter-pick` | `attack_move,attack_unit,build,move_units,place_building,stop` |
| `adversarial-duel` ⚠️ | `(empty)` |
| `artofwar-indirect-approach` | `attack_unit,move_units,stop` |
| `artofwar-lure-the-tiger` | `attack_unit,move_units,stop` |
| `artofwar-sequenced-citadel` | `attack_unit,move_units,stop` |
| `harass-response-preserve` ⚠️ | `attack_move,attack_unit,move_units,stop` |
| `strict-production-bom` | `build,deploy,place_building,stop` |
| `strict-sequence` | `move_units,stop` |
| `strict-toolban-fidelity-under-pressure` | `move_units,observe,stop` |

⚠️ `adversarial-duel` exposes no tools at all — likely defers to an
inner sub-pack. Confirm.
⚠️ `adv-asymmetric-weaker-must-win` and `harass-response-preserve` are
the only adversarial packs without `build`/`place_building` — pure
tactical, confirm intent.


### Special / Superweapon (6)

`union: attack_unit, build, c4_detonate, capture_actor, fire_superweapon, infiltrate, move_units, observe, place_building, set_stance, stop`

| Pack | Tools |
|---|---|
| `power-budget-online` | `build,observe,place_building,stop` |
| `spec-engineer-capture` | `attack_unit,capture_actor,move_units,observe` |
| `spec-nuke-strike` | `fire_superweapon,observe` |
| `spec-spy-infiltrate` | `attack_unit,infiltrate,move_units,observe` |
| `spec-tanya-c4-strike` | `attack_unit,c4_detonate,move_units,observe,set_stance` |
| `spec-thief-steal-cash` | `attack_unit,infiltrate,move_units,observe` |


### Misc (2)

`union: attack_unit, move_units, stop, stop_units`

| Pack | Tools |
|---|---|
| `TEMPLATE` | `attack_unit,move_units,stop_units` |
| `custom-map-no-enemy` | `move_units,stop` |


## Audit walk plan

For each category, the human audit pass checks per pack:

1. **Objective text** is plain-English readable — does the briefing
   make the win condition obvious without RTS jargon decoding?
2. **`tools:`** matches the named skill — no missing verb
   (`mfb-base-1-defend-base-2-build` style), no extra verb that lets
   the model cheat the predicate.
3. **`base_map:` / per-level `base_map:`** resolves to a real
   `.oramap` (210/210 confirmed by the map audit; spot-check during
   the walk).
4. **Win / lose predicate** is reachable AND solvable under the
   declared tool allowlist (the `proc-only-defend-no-attack` style is
   the canonical edge case).
5. **Level matrix** — easy / medium / hard tiers exist where the pack
   declares them, and the difficulty knob is a real lever (turns,
   enemy count, fog, deadline) not a no-op.
6. **Discriminations** — stall / one-tool-only / cheat-the-predicate
   policies all reach the documented outcome (no defects, no draws on
   what should be a real LOSS).

Recommended order (smallest → largest, so we exercise the audit
process before the big buckets):

1. Misc (2)
2. Special / Superweapon (6)
3. Adversarial (10)
4. Coordination (11)
5. Tech / Decision (11)
6. Tempo / Timing (13)
7. Long-horizon (14)
8. Scouting (19)
9. Combat (25)
10. Defense (29)
11. Build / Construction (32)
12. Economy (38)
