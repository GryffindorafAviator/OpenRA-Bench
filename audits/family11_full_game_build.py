"""Builds audits/family11_full_game.csv — Family-11 (Full-Game Sequence)
manual audit.

One row per (pack, level). Family-11 is the integration test that combines
every other family's verb-set: plan → cash → build → produce vertical
force → attack. Most packs MUST have offensive enemies that escalate by
tier (§69).

F11-specific columns extend F1's base set:
  - enemy_threat_schedule  — wave timing per tier
  - production_buildings_required — buildings the agent must BUILD
  - force_composition_required — minimum vertical force per win clause
  - wrong_arm_traps — 2-3 documented losing single-arm policies

See audits/EDIT_PRINCIPLES_FAMILY11.md for the binding rules. Briefings
are SELF-CONTAINED (F1 §1), three-part (framing → forces → objective +
constraint), plain-English (F1 §3), with NO solution-leak (F1 §9.5 + F2
§18) — i.e. NO build-order prescription, NO arm choice telegraphed by
verb, NO per-policy outcome table.

The 7 packs in this family:
  1. f11-vertical-strike-ground-air — combined ground + air strike
  2. f11-vertical-strike-naval — ground + naval (coastal map)
  3. f11-full-combined-arms — all three arms, largest map
  4. f11-econ-tech-army-strike — pure 4-phase full-game arc
  5. f11-defense-then-counter — survive opening rush, then build + counter
  6. f11-pivot-on-scout — scout enemy arm, pick RPS counter (re-pivot on hard)
  7. f11-rebuild-after-attrition — scheduled-event wrecks base, rebuild + finish
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent / 'family11_full_game.csv'
R = []


def add(pack, level, cap, map_name, map_size, map_fit, tools, agent, enemy,
        threat, briefing, win, lose, max_turns, tick_budget,
        prod_buildings, force_comp, wrong_arm_traps):
    R.append(dict(
        pack=pack, level=level, capability=cap, map_name=map_name,
        map_size=map_size, map_fit=map_fit, tools=tools,
        agent_force=agent, enemy_force=enemy,
        enemy_threat_schedule=threat,
        briefing_RA=briefing, win_condition=win, lose_condition=lose,
        max_turns=max_turns, tick_budget=tick_budget,
        production_buildings_required=prod_buildings,
        force_composition_required=force_comp,
        wrong_arm_traps=wrong_arm_traps,
    ))


# Common tools palette for F11 (the full-game palette: build + place + move + attack + stop + scout-helpers)
F11_TOOLS = 'observe, build, place_building, move_units, attack_unit, attack_move, set_stance, stop'


# =============================================================================
# 1. f11-vertical-strike-ground-air — combined ground + air strike
# =============================================================================
P = 'f11-vertical-strike-ground-air'; C = 'reasoning'

add(P, 'easy', C, 'f11-vertical-strike-ground-air-arena', '96x40', 'wide-justified', F11_TOOLS,
    "Western base @(8,20) — Construction Yard, Power Plant, Ore Refinery and 1 starter harvester (unarmed ore-collector truck). $4500 starting cash.",
    "Enemy base @(85,20) — Construction Yard + Power Plant + 1 rifleman picket (Defend stance). Light static garrison; sentinel fact @(94,38).",
    "static garrison only (easy = no scheduled waves; the test is the build-then-strike chain itself)",
    "Commander, this is a vertical-strike rehearsal: you must build BOTH a ground production line AND an air production pad, field at least two medium tanks AND one helicopter, AND raze at least two enemy buildings — all inside about 90 turns. Your western base at (8,20) has a Construction Yard, Power Plant, Ore Refinery and one starter harvester (an unarmed ore-collector truck). Cash on hand: 4500. An enemy base sits at (85,20), 77 cells east, lightly guarded by a single rifleman. The chain must happen in order: production buildings up before units are counted; units up before the kills are counted.",
    "Build weap (War Factory) AND hpad (Helipad), field ≥2 medium tanks AND ≥1 helicopter, raze ≥2 enemy buildings, within 8000 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (8001 ticks).",
    90, 8103,
    "weap, hpad",
    "≥2× 2tnk + ≥1× heli",
    "all-ground LOSES (kill bar requires hpad+heli production gate, which never latches); all-air LOSES (kill bar requires weap+2tnk production gate); stall LOSES (deadline; no production)")
add(P, 'medium', C, 'f11-vertical-strike-ground-air-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Western base @(8,20) — Construction Yard, Power Plant, Ore Refinery and 2 starter harvesters. $5500 starting cash.",
    "Enemy base @(100,20) — Construction Yard + Power Plant + Barracks + 2× rifle + 1× rocket-soldier picket (Defend); sentinel fact @(110,38). Scheduled hunt-tank wave (3× medium tank) @ tick 1500.",
    "static garrison + scheduled_events hunt-tank wave (3× 2tnk) @ tick 1500 (medium escalation)",
    "Commander, this is a vertical strike against a heavier enemy garrison with a mid-game hunt-tank raid. You must build BOTH a war factory AND a helipad, field at least three medium tanks AND one helicopter, AND raze at least two enemy buildings, all inside about 110 turns. Your western base at (8,20) holds a Construction Yard, Power Plant, Ore Refinery and two starter harvesters. Cash on hand: 5500. An enemy base sits at (100,20), 92 cells east, defended by riflemen, a rocket soldier, and a hunt-tank raid that arrives mid-game around tick 1500. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND hpad, field ≥3 medium tanks AND ≥1 helicopter, raze ≥2 enemy buildings, within 9900 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (9901 ticks).",
    110, 9903,
    "weap, hpad, fix",
    "≥3× 2tnk + ≥1× heli",
    "all-ground LOSES (no air production); all-air LOSES (no ground production AND the scheduled tank raid overruns base without ground defense); stall LOSES (deadline + raid breach)")
add(P, 'hard', C, 'f11-vertical-strike-ground-air-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Seed-rotated western base: sp0 NORTH @(8,12) or sp1 SOUTH @(8,28). Each spawn: Construction Yard + Power Plant + Ore Refinery + 2 starter harvesters. $6500 starting cash.",
    "Enemy base @(100,20) — Construction Yard + Power Plant + Barracks + Helipad-marker + 3× rifle + 2× rocket-soldier picket + 1× anti-air (Defend); sentinel fact @(110,38). Scheduled waves: hunt-tank (3× 2tnk) @ tick 1500; rocket-soldier rush (5× e3) @ tick 3000; hunt-heli (2× heli) @ tick 4500.",
    "static garrison + scheduled_events: tank wave @ tick 1500 + rocket rush @ tick 3000 + heli intercept @ tick 4500 (hard tri-wave + enemy AA presence)",
    "Commander, this is a vertical strike against a defended enemy base under continuous waves. You must build BOTH a war factory AND a helipad, field at least three medium tanks AND at least two helicopters, AND raze at least three enemy buildings, all inside about 140 turns. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed and holds a Construction Yard, Power Plant, Ore Refinery and two starter harvesters. Cash on hand: 6500. An enemy base sits at (100,20), defended by riflemen, rocket soldiers, an anti-air emplacement, and three timed waves — hunt-tanks around tick 1500, a rocket rush around tick 3000, helicopter interceptors around tick 4500. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND hpad, field ≥3 medium tanks AND ≥2 helicopters, raze ≥3 enemy buildings, within 12500 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (12501 ticks).",
    140, 12513,
    "weap, hpad, fix",
    "≥3× 2tnk + ≥2× heli",
    "all-ground LOSES (raid at tick 4500 is heli — only own heli can intercept; AND no hpad gate); all-air LOSES (enemy AA shreds heli without ground escort + scheduled tank raid breaches base); stall LOSES (waves overrun); single-heli LOSES (kill bar requires ≥2 heli surviving the AA)")


# =============================================================================
# 2. f11-vertical-strike-naval — ground + naval (light air maybe), coastal map
# =============================================================================
P = 'f11-vertical-strike-naval'; C = 'reasoning'

add(P, 'easy', C, 'f11-vertical-strike-naval-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Western coastal base @(10,20) — Construction Yard, Power Plant, Ore Refinery and 1 starter harvester. $4500 starting cash. Water channel along x=22..24 (3-cell-wide vertical band).",
    "Enemy positions: coastal building (Barracks) @(28,8) standing two cells inland of the east shoreline; sentinel fact @(108,38). Light static defense (1× rifleman @ coast).",
    "static garrison only (easy = no waves; the test is the dual-arm chain — ground inland + naval shore-strike)",
    "Commander, this is a coastal vertical strike. You must build BOTH a war factory AND a shipyard, field at least two medium tanks AND one destroyer (a naval warship that floats on water and cannot move onto land), AND raze at least two enemy buildings, all inside about 100 turns. Your western base at (10,20) sits west of a vertical water channel running from x=22 to x=24. Cash on hand: 4500. The enemy has a coastal Barracks at (28,8) just east of the shoreline (a destroyer's gun reaches it from the water) AND a deep-inland Construction Yard further east, beyond any naval range. A shipyard must be built ADJACENT to a water cell to function. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND syrd (Shipyard), field ≥2 medium tanks AND ≥1 destroyer, raze ≥2 enemy buildings (coastal AND inland), within 9000 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (9001 ticks).",
    100, 9003,
    "weap, syrd",
    "≥2× 2tnk + ≥1× dd",
    "all-ground LOSES (kill bar requires syrd+dd production gate; never latches); all-navy LOSES (destroyer can't range deep-inland building — kill clause requires BOTH coastal AND inland razed); stall LOSES (deadline)")
add(P, 'medium', C, 'f11-vertical-strike-naval-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Western coastal base @(10,20) — Construction Yard, Power Plant, Ore Refinery and 2 starter harvesters. $5500 starting cash. Water channel x=22..24.",
    "Enemy positions: coastal Barracks @(28,8) and coastal Power Plant @(28,32) on the east shoreline; deep-inland Construction Yard @(95,20); sentinel fact @(108,38). Static defense (1× rocket @ each coastal building, 2× rifle @ inland base). Scheduled enemy-rusher wave (2× medium tank) @ tick 1800 from the inland base.",
    "static garrison + scheduled_events rusher (2× 2tnk) @ tick 1800 (medium escalation; tanks closing from the east overland)",
    "Commander, this is a coastal vertical strike against a heavier garrison with a mid-game tank raid. You must build BOTH a war factory AND a shipyard, field at least three medium tanks AND one destroyer, AND raze at least three enemy buildings, all inside about 130 turns. Your western base at (10,20) sits west of a vertical water channel from x=22 to x=24. Cash on hand: 5500. The enemy holds two coastal buildings just east of the shoreline (in destroyer range) and a deep-inland Construction Yard at (95,20) past any naval range. A tank raid lands around tick 1800. A shipyard must be built ADJACENT to a water cell to function. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND syrd, field ≥3 medium tanks AND ≥1 destroyer, raze ≥3 enemy buildings (≥1 coastal + ≥1 inland), within 11700 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (11701 ticks).",
    130, 11703,
    "weap, syrd, fix",
    "≥3× 2tnk + ≥1× dd",
    "all-ground LOSES (no syrd gate AND tank raid plus no ranged shore-strike means coastal buildings hard to crack); all-navy LOSES (can't kill inland AND the overland tank raid breaches base without ground defense); stall LOSES (deadline + raid breach)")
add(P, 'hard', C, 'f11-vertical-strike-naval-arena', '128x40', 'wide-justified', F11_TOOLS,
    "Seed-rotated western coastal base: sp0 NORTH @(10,12) or sp1 SOUTH @(10,28). Each spawn: Construction Yard + Power Plant + Ore Refinery + 2 starter harvesters. $6500 starting cash. Water channel x=22..25 (4-cell band).",
    "Enemy positions: coastal Barracks @(29,8), coastal Power Plant @(29,32), coastal Defense Turret (gun) @(29,20), deep-inland Construction Yard @(110,20), inland Service Depot @(110,12); sentinel fact @(124,38). Picket: 2× rocket @ each coastal building, 3× rifle + 1× rocket @ inland. Scheduled waves: rusher tanks (3× 2tnk) @ tick 1800; rocket-soldier rush (4× e3) @ tick 3500.",
    "static garrison incl. coastal AA-capable turret + scheduled_events: tank rusher @ tick 1800 + rocket rush @ tick 3500 (hard dual-wave + coastal turret menaces dd in range)",
    "Commander, this is a coastal vertical strike against a defended enemy base under timed waves. You must build BOTH a war factory AND a shipyard, field at least three medium tanks AND at least two destroyers, AND raze at least four enemy buildings (with at least one coastal AND at least one deep-inland), all inside about 160 turns. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed and sits west of a 4-cell water channel from x=22 to x=25. Cash on hand: 6500. The enemy holds three coastal buildings (one is a defense turret that fires on naval targets in range), two deep-inland buildings past naval reach, and lands two timed waves — a tank rusher around tick 1800 and a rocket rush around tick 3500. A shipyard must be built ADJACENT to a water cell. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND syrd, field ≥3 medium tanks AND ≥2 destroyers, raze ≥4 enemy buildings (≥1 coastal + ≥1 inland), within 14500 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (14501 ticks).",
    160, 14403,
    "weap, syrd, fix",
    "≥3× 2tnk + ≥2× dd",
    "all-ground LOSES (no naval gate + coastal turret denies overland approach near the shore); all-navy LOSES (inland buildings outside range AND the inland-tank raid overruns base without ground defense); single-dd LOSES (coastal defense turret kills a lone destroyer before kill bar latches); stall LOSES (waves overrun)")


# =============================================================================
# 3. f11-full-combined-arms — all three arms, largest map
# =============================================================================
P = 'f11-full-combined-arms'; C = 'reasoning'

add(P, 'easy', C, 'f11-full-combined-arms-arena', '128x40', 'wide-justified', F11_TOOLS,
    "Western coastal base @(10,20) — Construction Yard, Power Plant, Ore Refinery and 2 starter harvesters. $6000 starting cash. Water channel x=22..24 (3-cell band).",
    "Three enemy positions: coastal Barracks @(28,8); midfield Power Plant @(70,20); deep-inland Construction Yard @(115,20). Sentinel fact @(125,38). Light static defense (1× rifleman per building).",
    "static garrison only (easy = no waves; the test is the three-arm chain itself)",
    "Commander, this is a full combined-arms strike. You must build a war factory, a helipad, AND a shipyard; field at least one medium tank, one helicopter, AND one destroyer; AND raze at least three enemy buildings, all inside about 130 turns. Your western base at (10,20) sits west of a 3-cell water channel from x=22 to x=24. Cash on hand: 6000. The enemy occupies three positions spread east: a coastal Barracks at (28,8), a midfield Power Plant at (70,20), and a deep-inland Construction Yard at (115,20). Each is lightly guarded. A shipyard must be built ADJACENT to a water cell. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND hpad AND syrd, field ≥1 medium tank AND ≥1 helicopter AND ≥1 destroyer, raze ≥3 enemy buildings, within 11700 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (11701 ticks).",
    130, 11703,
    "weap, hpad, syrd",
    "≥1× 2tnk + ≥1× heli + ≥1× dd",
    "single-arm LOSES (no matter which arm; kill bar requires THREE production gates AND THREE arm types in force); stall LOSES (deadline)")
add(P, 'medium', C, 'f11-full-combined-arms-arena', '128x40', 'wide-justified', F11_TOOLS,
    "Western coastal base @(10,20) — Construction Yard, Power Plant, Ore Refinery and 2 starter harvesters. $7000 starting cash. Water channel x=22..24.",
    "Three enemy positions: coastal Barracks @(28,8) + 1× rocket; midfield Power Plant @(70,20) + 1× rocket + 1× anti-air; deep-inland Construction Yard @(115,20) + 2× rifle. Sentinel fact @(125,38). Scheduled mixed wave (2× 2tnk + 1× heli) @ tick 2000.",
    "static garrison + mid-tier AA + scheduled_events mixed wave (2× 2tnk + 1× heli) @ tick 2000 (medium escalation; cross-arm pressure)",
    "Commander, this is a full combined-arms strike against a defended enemy spread with a mid-game mixed raid. You must build a war factory, a helipad, AND a shipyard; field at least two medium tanks, one helicopter, AND one destroyer; AND raze at least three enemy buildings, all inside about 150 turns. Your western base at (10,20) sits west of a 3-cell water channel from x=22 to x=24. Cash on hand: 7000. The enemy holds a coastal Barracks at (28,8), a midfield Power Plant at (70,20) with an anti-air emplacement, and a deep-inland Construction Yard at (115,20). A mixed tank-and-helicopter raid lands around tick 2000. A shipyard must be built ADJACENT to a water cell. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND hpad AND syrd, field ≥2 medium tanks AND ≥1 helicopter AND ≥1 destroyer, raze ≥3 enemy buildings, within 13500 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (13501 ticks).",
    150, 13503,
    "weap, hpad, syrd, fix",
    "≥2× 2tnk + ≥1× heli + ≥1× dd",
    "all-ground LOSES (kill bar needs hpad+syrd gates); all-air LOSES (midfield AA + mixed wave heli; AND no syrd gate); all-navy LOSES (midfield + inland out of range); single-arm + 1 token = same — only true three-arm builds win; stall LOSES (raid breach)")
add(P, 'hard', C, 'f11-full-combined-arms-arena', '128x48', 'wide-justified', F11_TOOLS,
    "Seed-rotated western coastal base: sp0 NORTH @(10,16) or sp1 SOUTH @(10,32). Each spawn: Construction Yard + Power Plant + Ore Refinery + 2 starter harvesters. $8000 starting cash. Water channel x=22..25 (4-cell band).",
    "Four enemy positions: coastal Barracks @(29,10) + 2× rocket; coastal AA turret (gun) @(29,38); midfield Service Depot @(70,24) + 1× rocket + 1× anti-air; deep-inland Construction Yard @(115,16) + deep-inland Power Plant @(115,32) + 3× rifle + 1× rocket. Sentinel fact @(126,46). Scheduled waves: mixed (3× 2tnk + 1× heli) @ tick 2000; coastal raid (2× dd from enemy syrd) @ tick 3500; rocket rush (5× e3) @ tick 5000.",
    "static garrison incl. coastal AA + midfield AA + scheduled_events tri-wave: mixed @ 2000 + naval raid @ 3500 + rocket rush @ 5000 (hard; all three threat domains live concurrently)",
    "Commander, this is a full combined-arms strike against a defended enemy spread under continuous waves. You must build a war factory, a helipad, AND a shipyard; field at least three medium tanks, two helicopters, AND two destroyers; AND raze at least four enemy buildings, all inside about 180 turns. Your western base spawns NORTH (y=16) or SOUTH (y=32) by seed and sits west of a 4-cell water channel from x=22 to x=25. Cash on hand: 8000. The enemy holds two coastal positions (one is a defense turret), a midfield Service Depot with anti-air, and two deep-inland buildings. Three waves arrive — mixed armour-and-helicopter around tick 2000, a coastal naval raid around tick 3500, and a rocket rush around tick 5000. A shipyard must be built ADJACENT to a water cell. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND hpad AND syrd, field ≥3 medium tanks AND ≥2 helicopters AND ≥2 destroyers, raze ≥4 enemy buildings, within 16000 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (16001 ticks).",
    180, 16203,
    "weap, hpad, syrd, fix",
    "≥3× 2tnk + ≥2× heli + ≥2× dd",
    "any single-arm LOSES (one of the three production gates never latches); two-arm LOSES (the third arm gate never latches; AND the missing-arm-coverage wave overruns base); stall LOSES (tri-wave overrun)")


# =============================================================================
# 4. f11-econ-tech-army-strike — pure full-game arc, ground primary
# =============================================================================
P = 'f11-econ-tech-army-strike'; C = 'reasoning'

add(P, 'easy', C, 'f11-econ-tech-army-strike-arena', '96x40', 'fit', F11_TOOLS,
    "Western base @(8,20) — Construction Yard, Power Plant, Ore Refinery and 1 starter harvester. $3500 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Power Plant + 1× rifleman picket (Defend stance); sentinel fact @(92,38).",
    "light static garrison only (easy = no scheduled waves; tests pure 4-phase econ→tech→army→strike arc)",
    "Commander, this is a four-phase operational arc: econ to fund, tech to enable, army to mass, strike to win. You are given a western base at (8,20) — Construction Yard, Power Plant, Ore Refinery and one starter harvester. Cash on hand: 3500. The enemy holds a Construction Yard at (82,20), 74 cells east, guarded by a single rifleman. Win bar: stand up the production chain, field at least four medium tanks, AND raze the enemy Construction Yard, all inside about 100 turns. The chain must happen in order: production buildings before units, units before the strike.",
    "Build weap, field ≥4 medium tanks, raze enemy Construction Yard, within 9000 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (9001 ticks).",
    100, 9003,
    "weap, fix",
    "≥4× 2tnk",
    "over-econ (≥3 procs, no army) LOSES (cash piles, no production gate); over-tech (weap + fix + dome but no tanks) LOSES (army clause never latches); stall LOSES (deadline)")
add(P, 'medium', C, 'f11-econ-tech-army-strike-arena', '112x40', 'fit', F11_TOOLS,
    "Western base @(8,20) — Construction Yard, Power Plant, Ore Refinery and 1 starter harvester. $4000 starting cash.",
    "Enemy base @(98,20) — Construction Yard + Power Plant + Barracks + 2× rifle + 1× rocket-soldier picket (Defend); sentinel fact @(108,38). Scheduled raid (3× medium tank, hunt bot) @ tick 1500.",
    "static garrison + scheduled_events hunt raid (3× 2tnk) @ tick 1500 (medium escalation)",
    "Commander, this is a four-phase operational arc against a defended enemy with a mid-game raid. You are given a western base at (8,20) — Construction Yard, Power Plant, Ore Refinery and one starter harvester. Cash on hand: 4000. The enemy holds a defended base at (98,20), 90 cells east, with riflemen, a rocket soldier, and a hunt-tank raid that arrives around tick 1500. Win bar: stand up the production chain, field at least five medium tanks, AND raze at least two enemy buildings, all inside about 125 turns. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND fix, field ≥5 medium tanks, raze ≥2 enemy buildings, within 11250 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (11251 ticks).",
    125, 11253,
    "weap, fix",
    "≥5× 2tnk",
    "over-econ LOSES (raid at tick 1500 breaches a base with too few defenders); under-econ (single proc + no expansion harv) LOSES (cash starves; 5-tank army never affordable inside the clock); stall LOSES (raid + deadline)")
add(P, 'hard', C, 'f11-econ-tech-army-strike-arena', '112x40', 'fit', F11_TOOLS,
    "Seed-rotated western base: sp0 NORTH @(8,12) or sp1 SOUTH @(8,28). Each spawn: Construction Yard + Power Plant + Ore Refinery + 1 starter harvester. $4500 starting cash.",
    "Enemy base @(98,20) — Construction Yard + Power Plant + Barracks + War Factory-marker + 3× rifle + 2× rocket + 1× medium tank picket (Defend); sentinel fact @(108,38). Scheduled waves: hunt raid (3× 2tnk) @ tick 1500; main attack (5× e3 + 2× 2tnk hunt) @ tick 3000.",
    "static garrison + scheduled_events dual raid: hunt @ 1500 + main attack @ 3000 (hard escalation)",
    "Commander, this is a four-phase operational arc against a defended enemy under timed waves. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed and holds a Construction Yard, Power Plant, Ore Refinery and one starter harvester. Cash on hand: 4500. The enemy holds a defended base at (98,20) with riflemen, rocket soldiers, a medium tank, and two timed raids — a hunt-tank raid around tick 1500 and a main attack of rocket soldiers and tanks around tick 3000. Win bar: stand up the production chain, field at least six medium tanks, AND raze at least three enemy buildings, all inside about 150 turns. The chain must happen in order: production buildings before units, units before kills.",
    "Build weap AND fix, field ≥6 medium tanks, raze ≥3 enemy buildings, within 13500 ticks. Western Construction Yard must survive.",
    "Western Construction Yard destroyed, all units dead, or deadline (13501 ticks).",
    150, 13503,
    "weap, fix",
    "≥6× 2tnk",
    "over-econ LOSES (raids overrun under-defended base); under-econ LOSES (6-tank army unaffordable on single-harv income); stall LOSES; rush-w-starter (no production) LOSES (no starters — pre-placed force is empty)")


# =============================================================================
# 5. f11-defense-then-counter — survive opening rush, then build + counter
# =============================================================================
P = 'f11-defense-then-counter'; C = 'reasoning'

add(P, 'easy', C, 'f11-defense-then-counter-arena', '96x40', 'fit', F11_TOOLS,
    "Western base @(12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, 1 starter harvester, 2× pre-placed rifleman (Defend stance) on the eastern face @(18,18-22). $3500 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Barracks + 3× medium tank pre-positioned @(45,18-22) closing west (hunt bot, st3). Sentinel fact @(92,38). Scheduled secondary wave (3× e3 rocket soldiers, hunt) @ tick 1500.",
    "OFFENSIVE FROM t=0: 3× 2tnk hunt rush already moving (tick 0-500) + scheduled_events rocket rush @ tick 1500 (defense-then-counter: survive the opener, then build)",
    "Commander, this is a defense-then-counter operation. You are given a western base at (12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, one starter harvester, and two pre-placed riflemen on the eastern face. Cash on hand: 3500. Three enemy medium tanks at (45, 18-22) are already closing west on a hunt order; a follow-up rocket-soldier rush arrives around tick 1500. The enemy Construction Yard sits at (82,20), 70 cells east. Win bar: SURVIVE the opening with the base intact and at least one harvester alive, BUILD a war factory, field at least four medium tanks, AND raze at least one enemy production building, all inside about 100 turns. The chain must happen in order: survival before production, production before counter-force, counter before kill.",
    "Construction Yard alive AND ≥1 harvester alive (after tick 1500), build weap, field ≥4 medium tanks, raze ≥1 enemy production building, within 9000 ticks.",
    "Construction Yard destroyed, all harvesters dead, or deadline (9001 ticks).",
    100, 9003,
    "weap",
    "≥4× 2tnk",
    "panic-build-army-too-soon (skip defense, the tank rush breaches base before army is ready) LOSES; over-defense (build 4 pbox, no weap, no counter) LOSES (counter clause never latches); stall LOSES (rush breach)")
add(P, 'medium', C, 'f11-defense-then-counter-arena', '96x40', 'fit', F11_TOOLS,
    "Western base @(12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, 1 starter harvester, 3× pre-placed rifleman @(18,18-22). $4000 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Barracks + 4× medium tank pre-positioned @(45,18-22) closing west (hunt). Sentinel fact @(92,38). Scheduled waves: rocket rush (5× e3, hunt) @ tick 1500; second tank wave (3× 2tnk, hunt) @ tick 3000.",
    "OFFENSIVE FROM t=0: 4× 2tnk hunt + scheduled_events rocket rush @ 1500 + second tank wave @ 3000 (medium escalation; sustained pressure)",
    "Commander, this is a defense-then-counter operation against sustained pressure. You are given a western base at (12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, one starter harvester, three pre-placed riflemen. Cash on hand: 4000. Four enemy medium tanks at (45, 18-22) are closing west on a hunt order; a rocket-soldier rush arrives around tick 1500 and a second tank wave around tick 3000. The enemy Construction Yard sits at (82,20). Win bar: SURVIVE through the second wave with the base intact and at least one harvester alive, BUILD a war factory, field at least five medium tanks, AND raze at least one enemy production building, all inside about 130 turns. The chain must happen in order.",
    "Construction Yard alive AND ≥1 harvester alive (after tick 3000), build weap, field ≥5 medium tanks, raze ≥1 enemy production building, within 11700 ticks.",
    "Construction Yard destroyed, all harvesters dead, or deadline (11701 ticks).",
    130, 11703,
    "weap, fix",
    "≥5× 2tnk",
    "panic-build (skip defense) LOSES (4-tank hunt + rocket wave breach base inside 1500 ticks); over-defense (8× pbox no weap) LOSES (counter never fires); under-defense (just the starters) LOSES (second tank wave finishes the base); stall LOSES")
add(P, 'hard', C, 'f11-defense-then-counter-arena', '96x40', 'fit', F11_TOOLS,
    "Seed-rotated western base: sp0 NORTH @(12,12) or sp1 SOUTH @(12,28). Each spawn: Construction Yard + Power Plant + Ore Refinery + Barracks + 1 starter harvester + 3× pre-placed rifleman. $4500 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Barracks + 5× medium tank pre-positioned @(45,12-28) closing west (hunt). Sentinel fact @(92,38). Scheduled waves: rocket rush (6× e3, hunt) @ tick 1500; second tank wave (4× 2tnk, hunt) @ tick 3000; third heli wave (2× heli, hunt) @ tick 4500.",
    "OFFENSIVE FROM t=0: 5× 2tnk hunt + scheduled_events tri-wave: rocket @ 1500 + tank wave @ 3000 + heli wave @ 4500 (hard sustained tri-domain pressure)",
    "Commander, this is a defense-then-counter operation under sustained tri-domain pressure. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed and holds a Construction Yard, Power Plant, Ore Refinery, Barracks, one harvester, and three riflemen. Cash on hand: 4500. Five enemy medium tanks are closing west on a hunt order; a rocket rush arrives around tick 1500, a second tank wave around tick 3000, and a helicopter wave around tick 4500. The enemy Construction Yard sits at (82,20). Win bar: SURVIVE through the third wave with the base intact and at least one harvester alive, BUILD a war factory AND a helipad, field at least four medium tanks AND at least one helicopter, AND raze at least one enemy production building, all inside about 160 turns. The chain must happen in order.",
    "Construction Yard alive AND ≥1 harvester alive (after tick 4500), build weap AND hpad, field ≥4 medium tanks AND ≥1 helicopter, raze ≥1 enemy production building, within 14400 ticks.",
    "Construction Yard destroyed, all harvesters dead, or deadline (14401 ticks).",
    160, 14403,
    "weap, hpad, fix",
    "≥4× 2tnk + ≥1× heli",
    "panic-build LOSES (opener breach); over-defense LOSES (no counter); ground-only LOSES (heli wave at tick 4500 destroys any base without aa or own air); stall LOSES")


# =============================================================================
# 6. f11-pivot-on-scout — scout enemy arm, pick RPS counter (re-pivot on hard)
# =============================================================================
P = 'f11-pivot-on-scout'; C = 'reasoning'

add(P, 'easy', C, 'f11-pivot-on-scout-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Western base @(8,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, 1 starter harvester, 1 starter jeep (scout, fast, light armament). $4500 starting cash. FOG-OF-WAR ON.",
    "Enemy base @(95,20) — Construction Yard + Barracks + ONE production building visible from the scout vantage @(80,20): on this seed, ALL-INFANTRY (Barracks active) — 4× rifleman + 2× rocket-soldier patrol pre-placed (Defend). Sentinel fact @(108,38).",
    "static + visible single arm (no scheduled waves at easy; the test is observe-the-arm + counter-build)",
    "Commander, this is a perception-and-counter mission against an enemy committed to a single arm. You are given a western base at (8,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, one starter harvester, and one starter jeep. Cash on hand: 4500. Fog of war is on. The enemy base sits at (95,20), 87 cells east. A forward scout vantage at around (80,20) can reveal which production building the enemy is running — they have committed to ONE arm (infantry, armour, or air). Win bar: build a war factory AND/OR a helipad, field at least three units of the COUNTER arm to whatever the enemy is fielding, AND raze at least one enemy production building, all inside about 110 turns. The chain must happen in order: scout the arm before producing the counter, then strike.",
    "Scout the (80,20) vantage cell (jeep enters), then build the appropriate production line (weap OR hpad), field ≥3 units of the counter arm, raze ≥1 enemy production building, within 9900 ticks.",
    "Construction Yard destroyed, all units dead, or deadline (9901 ticks).",
    110, 9903,
    "weap OR hpad (depending on scouted arm)",
    "≥3 units of correct counter arm",
    "same-arm-as-enemy LOSES (kill bar requires counter — same-arm bounces off the picket); ignored-scout (build before scouting) LOSES if guessed wrong on this seed; stall LOSES")
add(P, 'medium', C, 'f11-pivot-on-scout-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Western base @(8,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, 1 starter harvester, 1 starter jeep. $5000 starting cash. Fog on.",
    "Enemy base @(95,20) — Construction Yard + Barracks + TWO production buildings visible from the scout vantage @(80,20): the model must observe WHICH is set_primary (the active producer). On this seed: armour set_primary (4× 2tnk + 1× rocket-soldier picket). Sentinel fact @(108,38). Scheduled scout-spotter wave (1× jeep) @ tick 2000 (re-spawn to keep observation possible).",
    "static + two-building enemy with set_primary discrimination + scheduled scout-respawn (medium: ambiguity in which arm is producing)",
    "Commander, this is a perception-and-counter mission against an enemy with TWO production buildings, only one of which is active. You are given a western base at (8,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, one starter harvester, and one starter jeep. Cash on hand: 5000. Fog of war is on. The enemy base sits at (95,20). A forward scout vantage at around (80,20) can reveal which of the enemy's two production buildings is the active producer (the primary). Win bar: build a war factory AND/OR a helipad, field at least three units of the COUNTER arm to whatever the enemy is producing, AND raze at least two enemy buildings (including the active producer), all inside about 130 turns.",
    "Scout (80,20), build counter-arm production, field ≥3 units of the counter arm, raze ≥2 enemy buildings including the active production building, within 11700 ticks.",
    "Construction Yard destroyed, all units dead, or deadline (11701 ticks).",
    130, 11703,
    "weap OR hpad (the counter to the scouted active-producer)",
    "≥3 units of correct counter arm",
    "same-arm LOSES; ignored-scout LOSES (guessing wrong loses to deadline; trying to build both arms wastes cash and trips deadline); stall LOSES")
add(P, 'hard', C, 'f11-pivot-on-scout-arena', '112x40', 'wide-justified', F11_TOOLS,
    "Seed-rotated western base: sp0 NORTH @(8,12) or sp1 SOUTH @(8,28). Each spawn: Construction Yard + Power Plant + Ore Refinery + Barracks + 1 starter harvester + 1 starter jeep. $5500 starting cash. Fog on.",
    "Enemy base @(95,20) — Construction Yard + Barracks + TWO production buildings + arm switches mid-episode (initial arm visible from scout @(80,20); SWITCH via scheduled_events.spawn_actors @ tick 2000 — new arm spawns + previous arm picket withers in priority). Initial: 4× 2tnk picket; after tick 2000: 5× heli wave injected. Sentinel fact @(108,38).",
    "static + two-building + scheduled_events.spawn_actors at tick 2000 SWITCHES the arm (hard: re-scout + re-pivot under fog)",
    "Commander, this is a perception-and-counter mission where the enemy SWITCHES arms mid-game. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed and holds a Construction Yard, Power Plant, Ore Refinery, Barracks, one harvester, and one jeep. Cash on hand: 5500. Fog of war is on. The enemy base sits at (95,20). A forward scout vantage at around (80,20) reveals the enemy's initial production arm; around tick 2000 the enemy switches arms, and only a fresh scout move into the vantage will reveal the new arm. Win bar: scout the initial arm, then scout AGAIN after tick 2000, field at least three units of the COUNTER to the SECOND arm, AND raze at least two enemy buildings, all inside about 150 turns. The chain must happen in order: first scout before initial production, second scout after the switch, then counter and strike.",
    "Scout (80,20) before tick 2000, scout (80,20) AFTER tick 2000, build the counter-arm production for the SECOND arm, field ≥3 units of that counter, raze ≥2 enemy buildings, within 13500 ticks.",
    "Construction Yard destroyed, all units dead, or deadline (13501 ticks).",
    150, 13503,
    "weap AND/OR hpad (often must build BOTH if the second arm differs from the first)",
    "≥3 units of correct counter to SECOND arm",
    "locked-in (counter the first arm, ignore the switch) LOSES (second arm invalidates the counter); ignored-second-scout LOSES (guesses wrong); single-scout (only first) LOSES; stall LOSES")


# =============================================================================
# 7. f11-rebuild-after-attrition — scheduled-event wrecks base, rebuild + finish
# =============================================================================
P = 'f11-rebuild-after-attrition'; C = 'reasoning'

add(P, 'easy', C, 'f11-rebuild-after-attrition-arena', '96x40', 'fit', F11_TOOLS,
    "Western base @(12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, War Factory @(15,22), 2 starter harvesters, 2 starter medium tanks. $3000 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Barracks + 2× rifleman + 1× rocket-soldier picket (Defend); sentinel fact @(92,38). Scheduled scout drone (1× heli, hunt) @ tick 500. NO offensive enemy waves (the attrition IS the pressure).",
    "scheduled_events.destroy_actors @ tick 800 (wipes pre-placed war factory @(15,22) in radius:3) + scheduled scout drone @ tick 500 + termination.agent_units_killed: false",
    "Commander, this is a rebuild-after-attrition mission. You are given a western base at (12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, a War Factory at (15,22), two harvesters, and two starter medium tanks on hold-fire. Cash on hand: 3000. Around tick 800 a scripted strike will RAZE the War Factory (and only the war factory). You must rebuild it AND continue producing. The enemy holds a Construction Yard at (82,20), 70 cells east, lightly defended. Win bar: REBUILD the war factory after the attrition strike, field at least three medium tanks (AFTER the rebuild), AND raze at least one enemy production building, all inside about 110 turns. The chain must happen in order: survive the attrition, rebuild, produce, strike.",
    "After tick 1000: building_count_gte: weap, n: 1 (i.e. weap rebuilt) AND unit_type_count_gte: 2tnk, n: 3 AND raze ≥1 enemy production building. Within 9900 ticks.",
    "Construction Yard destroyed, all harvesters dead, no weap rebuilt by tick 4000, or deadline (9901 ticks).",
    110, 9903,
    "weap (rebuild after attrition)",
    "≥3× 2tnk (built AFTER rebuild)",
    "pre-attrition-only force (build to 4 tanks by tick 700, then ignore rebuild) LOSES (kill bar requires weap ALIVE post-attrition + new tanks); ignore-rebuild (just push with 2 starter tanks) LOSES (insufficient force, no production after wipe); stall LOSES")
add(P, 'medium', C, 'f11-rebuild-after-attrition-arena', '96x40', 'fit', F11_TOOLS,
    "Western base @(12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, War Factory @(15,22), Service Depot @(18,22), 2 starter harvesters, 2 starter medium tanks. $3500 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Barracks + 3× rifle + 2× rocket-soldier picket (Defend); sentinel fact @(92,38). Scheduled scout drone @ tick 500. Light enemy harassment: scheduled rocket-soldier raid (2× e3, hunt) @ tick 2500.",
    "scheduled_events.destroy_actors @ tick 800 (wipes weap) + scheduled scout drone @ 500 + light harass raid @ 2500 + termination.agent_units_killed: false (medium escalation; the rebuild is under modest harassment)",
    "Commander, this is a rebuild-after-attrition mission under modest harassment. You are given a western base at (12,20) — Construction Yard, Power Plant, Ore Refinery, Barracks, War Factory at (15,22), Service Depot at (18,22), two harvesters, and two starter medium tanks on hold-fire. Cash on hand: 3500. Around tick 800 a scripted strike will RAZE the War Factory. Around tick 2500 a small rocket-soldier raid arrives. You must rebuild AND continue producing AND defend. The enemy holds a defended base at (82,20). Win bar: REBUILD the war factory after attrition, field at least four medium tanks (after the rebuild), AND raze at least two enemy buildings, all inside about 130 turns. The chain must happen in order.",
    "After tick 1000: building_count_gte: weap, n: 1 AND unit_type_count_gte: 2tnk, n: 4 AND raze ≥2 enemy buildings. Within 11700 ticks.",
    "Construction Yard destroyed, all harvesters dead, no weap rebuilt by tick 4500, or deadline (11701 ticks).",
    130, 11703,
    "weap (rebuild)",
    "≥4× 2tnk (post-rebuild)",
    "pre-attrition-only LOSES; ignore-rebuild LOSES; stall LOSES; over-econ (rebuild proc instead of weap) LOSES (weap clause never relatches)")
add(P, 'hard', C, 'f11-rebuild-after-attrition-arena', '96x40', 'fit', F11_TOOLS,
    "Seed-rotated western base: sp0 NORTH @(12,12) or sp1 SOUTH @(12,28). Each spawn: Construction Yard + Power Plant + Ore Refinery + Barracks + War Factory + Service Depot + 2 starter harvesters + 2 starter medium tanks. $4000 starting cash.",
    "Enemy base @(82,20) — Construction Yard + Barracks + 4× rifle + 2× rocket-soldier + 1× medium tank picket (Defend); sentinel fact @(92,38). Scheduled scout drone @ 500. Scheduled rocket raid (3× e3, hunt) @ 2500.",
    "scheduled_events.destroy_actors TWICE: tick 800 wipes weap + tick 2000 wipes proc + scheduled scout drone @ 500 + harass raid @ 2500 + termination.agent_units_killed: false (hard: dual rebuild under sustained pressure)",
    "Commander, this is a rebuild-after-attrition mission against TWO scripted strikes. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed and holds a Construction Yard, Power Plant, Ore Refinery, Barracks, War Factory, Service Depot, two harvesters, and two starter medium tanks on hold-fire. Cash on hand: 4000. Around tick 800 a scripted strike will RAZE the War Factory; around tick 2000 a second strike will RAZE the Ore Refinery. Around tick 2500 a rocket-soldier raid arrives. The enemy holds a defended base at (82,20). Win bar: REBUILD BOTH the war factory AND the refinery, field at least five medium tanks (after both rebuilds), AND raze at least two enemy buildings, all inside about 160 turns. The chain must happen in order.",
    "After tick 2200: building_count_gte: weap, n: 1 AND building_count_gte: proc, n: 1 AND unit_type_count_gte: 2tnk, n: 5 AND raze ≥2 enemy buildings. Within 14400 ticks.",
    "Construction Yard destroyed, all harvesters dead, no weap or proc rebuilt by tick 5500, or deadline (14401 ticks).",
    160, 14403,
    "weap (rebuild), proc (rebuild)",
    "≥5× 2tnk (post-double-rebuild)",
    "pre-attrition-only LOSES (army gets stuck w/o weap or income post-tick-2000); ignore-rebuild LOSES; single-rebuild (weap only, no proc) LOSES (proc clause never relatches); stall LOSES")


# =============================================================================
# Emit CSV
# =============================================================================
fields = ['pack', 'level', 'capability', 'map_name', 'map_size', 'map_fit', 'tools',
          'agent_force', 'enemy_force', 'enemy_threat_schedule',
          'briefing_RA', 'win_condition', 'lose_condition',
          'max_turns', 'tick_budget',
          'production_buildings_required', 'force_composition_required',
          'wrong_arm_traps']

with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    w.writeheader()
    for r in R:
        w.writerow(r)

print(f'Wrote {len(R)} rows to {OUT}')
