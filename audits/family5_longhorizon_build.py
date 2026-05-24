"""Builds audits/family5_longhorizon.csv — Family-5 (Long-horizon) manual audit.

One row per (pack, level). Each briefing is SELF-CONTAINED in F1 officer
style — the model sees one level at a time, so every briefing fully
describes the mission framing, the forces given (with positions where
they matter), and the objective from scratch. No "same as before" or
"the same X" references.

Structure per briefing: mission framing -> what is given -> objective.
Red-Alert-specific terms (pillbox, stance, refinery, etc.) are explained
inline so non-RA readers can follow.

Family-5 specifics (per EDIT_PRINCIPLES_FAMILY5.md):
- Long-horizon packs run max_turns 50-160 with chained phases. Each
  briefing in the audit names the SITUATION + the WIN CLAUSES, never
  the per-phase RECIPE.
- The phase_chain column records the structure of the win predicate
  (e.g. "powr -> proc -> M=3500 -> weap -> kill 2" for a 5-stage
  PERT chain). The chain_idiom column flags `then`-strict /
  `all_of`-terminal / `all_of`-scheduled / hybrid composites.
- The leak_flags column records detected leak patterns in the YAML's
  current description ('per-phase-prescription', 'outcome-table',
  'order-spelled-out', 'clean').
- map_fit discipline carries forward from F1/F2/F3. 128x40 single-
  axis packs with the action confined to x<60 are large-trivial;
  160x60 multi-corner packs with load-bearing 120-cell separation are
  fit; the credit-only pack's long march IS the test, so fit despite
  the 160-cell map.

Scope (14 packs * 3 levels = 42 rows):
- 13 lh-* packs
- 1 longhorizon-* pack
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent / 'family5_longhorizon.csv'
R = []


def add(pack, level, cap, map_name, map_size, map_fit, tools, agent, enemy,
        briefing, win, lose, max_turns, tick_budget,
        posture='', posture_issue='',
        phase_chain='', chain_idiom='', leak_flags=''):
    R.append(dict(
        pack=pack, level=level, capability=cap, map_name=map_name,
        map_size=map_size, map_fit=map_fit, tools=tools,
        agent_force=agent, enemy_force=enemy,
        briefing_RA=briefing, win_condition=win, lose_condition=lose,
        max_turns=max_turns, tick_budget=tick_budget,
        enemy_posture=posture, posture_issue=posture_issue,
        phase_chain=phase_chain, chain_idiom=chain_idiom,
        leak_flags=leak_flags,
    ))


# == 1. lh-100-turn-marathon-survival - sustained ops over 90-120 turns
P = 'lh-100-turn-marathon-survival'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, repair, sell, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Construction Yard @(10,20) + Barracks + 2x Power Plant + Refinery + Harvester + 2 mines west + 4x rifle defenders @(16,20) st3',
    'Three staggered hunt-bot rifle pairs at x=60/80/100, y=20, st3; sentinel enemy fact @(120,20)',
    "Commander, this is a 90-turn endurance defence. Your western base — Construction Yard at (10,20), barracks, two power plants, ore refinery, one harvester with two mines just west, and four riflemen at (16,20) — must survive sustained pressure from three staggered enemy rifle squads at x=60, 80 and 100 closing in over the run. To win, at tick 7200 or later you must still own the yard, hold at least three of your own units alive, and have built at least two pillboxes (pillbox = stationary anti-infantry turret). Use the `repair` order to keep buildings standing under attrition. Within about 91 turns.",
    "Yard alive AND 3+ own units alive AND 2+ pillboxes built AND survived past tick 7200, within 8103 ticks.",
    "Yard destroyed, all own units dead, or deadline (tick 7200+ without all four win clauses).",
    90, 8103,
    'hunt st3; matches briefing — sustained pressure',
    '',
    'yard alive & 3+ units & 2+ pbox & after_ticks 7200', 'all_of-terminal',
    'outcome-table, order-spelled-out')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same yard + barracks + 2x powr + proc + harv + 4x rifle defenders',
    'Five staggered squads at x=60..115 (rifles + rocket soldiers mixed) st3; sentinel fact @(120,22)',
    "Commander, this is a 100-turn endurance defence. Same western base — Construction Yard at (10,20), barracks, two power plants, refinery, harvester, four starting riflemen — under heavier sustained pressure from FIVE staggered squads spanning x=60 to 115, mixing rifles and anti-tank rocket soldiers. To win, at tick 8100 or later you must still own the yard, hold at least three of your own units alive, and have built at least two pillboxes. Use `repair` to keep the pillbox line standing under the trickle of rocket-soldier damage. Within about 101 turns.",
    "Yard alive AND 3+ own units alive AND 2+ pillboxes built AND survived past tick 8100, within 9003 ticks.",
    "Yard destroyed, all own units dead, or deadline (tick 8100+ without all four win clauses).",
    100, 9003,
    'hunt st3; matches briefing',
    '',
    'yard alive & 3+ units & 2+ pbox & after_ticks 8100', 'all_of-terminal',
    'outcome-table, order-spelled-out')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=12) or SOUTH (y=28) by seed; same composition duplicated at each latitude',
    'Six staggered squads at TWO latitudes (north + south); enemy don\'t honour spawn_point so both bands always place',
    "Commander, this is a 120-turn endurance defence with seed-rotated staging. Your base — Construction Yard, barracks, two power plants, refinery, harvester, four riflemen plus two rocket soldiers — spawns from a NORTH latitude (y=12) OR SOUTH (y=28) depending on seed. Six staggered enemy squads close in from both latitudes; the on-latitude band converges on your active base first. At tick 9900 or later you must still own the yard, hold three units alive, and have built two pillboxes. A memorised pillbox cell fails on the flipped seed. Within about 121 turns.",
    "Yard alive AND 3+ own units alive AND 2+ pillboxes built AND survived past tick 9900, within 10803 ticks.",
    "Yard destroyed, all own units dead, or deadline (tick 9900+ without all four win clauses).",
    120, 10803,
    'hunt st3 from both latitudes; matches briefing',
    '',
    'yard alive & 3+ units & 2+ pbox & after_ticks 9900', 'all_of-terminal',
    'outcome-table, order-spelled-out')

# == 2. lh-build-army-coordinate-multifront-attack — mass then split
P = 'lh-build-army-coordinate-multifront-attack'; C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'arena 160x60', '160x60', 'fit', T,
    'Construction Yard @(10,30) + Barracks + Power Plant + War Factory + Service Depot + 2x medium tank starter @(16,28-30) HoldFire; 2 mines',
    'NE enemy yard @(130,15) + 2x rifle picket; SE enemy yard @(130,45) + 2x rifle picket; far-corner sentinel fact @(155,55)',
    "Commander, this is a two-front strike. Your western base — Construction Yard at (10,30), barracks, power plant, war factory, service depot, two starter medium tanks on hold-fire — faces TWO enemy headquarters on the east edge: one at NE (130,15) and one at SE (130,45), 120 cells apart vertically. Each is lightly defended by two riflemen. To win, you must field at least FOUR medium tanks total AND raze BOTH the NE yard AND the SE yard, in any order, before tick 7200. The phase markers are enforced in order — the army must be observed before either destruction counts. Within about 80 turns.",
    "Army >=4 tanks, THEN raze NE yard, THEN raze SE yard, all within 7200 ticks.",
    "Own yard destroyed, all own units dead, or deadline (7201 ticks).",
    80, 7203,
    'pickets stance:2 Defend; passive, intentional',
    '',
    'army >=4 -> raze NE -> raze SE', 'then-strict',
    'order-spelled-out')
add(P, 'medium', C, 'arena 160x60', '160x60', 'fit', T,
    'Same base + 2x starter tanks',
    'Same NE + SE yards each with 3-rifle picket; far-corner sentinel',
    "Commander, this is a two-front strike at heavier weight. Your western base at (10,30) — yard, barracks, power, war factory, depot, two starter medium tanks — faces two enemy yards at NE (130,15) and SE (130,45) each defended by a three-rifle picket. To win, field at least SIX medium tanks AND raze BOTH yards in any order before tick 7200. The order is enforced — assemble the army first. Within about 80 turns.",
    "Army >=6 tanks, THEN raze NE yard, THEN raze SE yard, all within 7200 ticks.",
    "Own yard destroyed, all own units dead, or deadline (7201 ticks).",
    80, 7203,
    'pickets stance:2 Defend; intentional',
    '',
    'army >=6 -> raze NE -> raze SE', 'then-strict',
    'order-spelled-out')
add(P, 'hard', C, 'arena 160x60', '160x60', 'fit', T,
    'Base spawns NORTH (y=18) or SOUTH (y=42) by seed; same composition',
    'NE (130,15) + SE (130,45) yards each with 1xe3 + 2xe1 picket; sentinel @(155,55)',
    "Commander, this is a two-front strike with seed-rotated staging. Your western base — yard, barracks, power, war factory, depot, two starter medium tanks — spawns NORTH (y=18) or SOUTH (y=42) by seed, so the near front flips between NE and SE. Both enemy yards (NE 130,15 and SE 130,45) are defended by one rocket soldier and two riflemen each. To win, field at least EIGHT medium tanks AND raze BOTH yards before tick 7200. A memorised near-front commit fails on the flipped seed. Within about 80 turns.",
    "Army >=8 tanks, THEN raze NE yard, THEN raze SE yard, all within 7200 ticks.",
    "Own yard destroyed, all own units dead, or deadline (7201 ticks).",
    80, 7203,
    'pickets stance:2 Defend; intentional',
    '',
    'army >=8 -> raze NE -> raze SE', 'then-strict',
    'order-spelled-out')

# == 3. lh-credit-only-final-phase — sparse-reward, one late objective
P = 'lh-credit-only-final-phase'; C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'arena 160x60', '160x60', 'fit', T,
    'Construction Yard @(10,30) + Barracks + Power Plant + War Factory + Service Depot + 2x medium tank @(16,28-30) HoldFire; 2 mines',
    '1x enemy yard @(130,30) + 2x rifle picket; far-corner sentinel @(155,55)',
    "Commander, this is a sparse-reward terminal-objective drill. Your western base at (10,30) — Construction Yard, barracks, power, war factory, service depot, two starter medium tanks on hold-fire — faces ONE enemy yard 120 cells east at (130,30), defended by two riflemen. Only the destruction of that far yard inside the deadline scores — building, massing and marching earn zero credit on their own. To win, raze the enemy yard before tick 7200. Within about 80 turns.",
    "Raze the enemy yard at (130,30) within 7200 ticks.",
    "Own yard destroyed, all own units dead, or deadline (7201 ticks).",
    80, 7203,
    'picket stance:2 Defend; intentional',
    '',
    'raze far yard (1 clause)', 'all_of-terminal',
    'outcome-table')
add(P, 'medium', C, 'arena 160x60', '160x60', 'fit', T,
    'Same base + 2x starter tanks',
    'Far yard + 4-unit picket (1xe3 + 3xe1); far-corner sentinel',
    "Commander, this is a sparse-reward terminal-objective drill at heavier weight. Same western base at (10,30) — yard, barracks, power, war factory, depot, two starter tanks. The far enemy yard at (130,30) is now defended by four units: one rocket soldier and three riflemen. Only razing it scores. Within about 80 turns.",
    "Raze the enemy yard at (130,30) within 7200 ticks.",
    "Own yard destroyed, all own units dead, or deadline (7201 ticks).",
    80, 7203,
    'picket stance:2 Defend; intentional',
    '',
    'raze far yard (1 clause)', 'all_of-terminal',
    'outcome-table')
add(P, 'hard', C, 'arena 160x60', '160x60', 'fit', T,
    'Base spawns NORTH (y=18) or SOUTH (y=42) by seed',
    'Same far yard (130,30) + 4-unit picket; sentinel',
    "Commander, this is a sparse-reward terminal-objective drill with seed-rotated staging. Your western base spawns NORTH (y=18) or SOUTH (y=42) by seed; the far enemy yard stays fixed at (130,30) so the diagonal march route flips per seed. Defending the yard are one rocket soldier and three riflemen. Only razing the yard scores. Within about 80 turns.",
    "Raze the enemy yard at (130,30) within 7200 ticks.",
    "Own yard destroyed, all own units dead, or deadline (7201 ticks).",
    80, 7203,
    'picket stance:2 Defend; intentional',
    '',
    'raze far yard (1 clause)', 'all_of-terminal',
    'outcome-table')

# == 4. lh-defense-tech-second-base — secure -> tech -> expand
P = 'lh-defense-tech-second-base'; C = 'reasoning'
T = 'observe, build, place_building, deploy, harvest, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'lh-defense-tech-second-base-arena', '160x60', 'fit', T,
    'Base #1 WEST: fact @(10,30) + tent + 2x powr + proc + harv + 3 rifle defenders st2; MCV @(40,30); 1 mine',
    'Light patrol: 2x e1 @(80,30) st3 (patrol bot); far-corner sentinel @(154,4)',
    "Commander, this is a secure-expand operation. Base #1 in the WEST — Construction Yard at (10,30), barracks, two power plants, refinery, harvester, three rifle defenders — plus a spare MCV (Mobile Construction Vehicle) staged at (40,30). A light Soviet patrol band of two riflemen wanders mid-map. The east target region is centred at (130,30), radius 8. To win, you must build at least TWO pillboxes at base #1, have a war factory standing, and deploy your MCV inside the eastern target region so a second Construction Yard stands there, with base #1 still alive — all before tick 8000. Within about 89 turns.",
    "2+ pbox at base#1, THEN war factory, THEN MCV deployed at (130,30) r=8, all within 8000 ticks AND base#1 yard alive.",
    "Base #1 yard destroyed, or deadline (8001 ticks).",
    90, 8103,
    'patrol bot st3; matches briefing',
    '',
    'pbox >=2 -> weap -> fact in east region', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'medium', C, 'lh-defense-tech-second-base-arena', '160x60', 'fit', T,
    'Same base #1 + MCV',
    'Heavier patrol: 3x e1 + 1x e3 @(80,30) st3; sentinel',
    "Commander, this is a secure-expand operation at heavier patrol weight. Same base #1 in the WEST — Construction Yard, barracks, two power plants, refinery, harvester, three defenders — plus the spare MCV. The patrol band is three riflemen plus one rocket soldier. The eastern target region is centred at (130,30) radius 8. To win, build at least THREE pillboxes at base #1, have a war factory standing, AND deploy the MCV inside the east region with base #1 alive, all before tick 7200. Within about 80 turns.",
    "3+ pbox at base#1, THEN war factory, THEN MCV deployed at (130,30) r=8, all within 7200 ticks AND base#1 yard alive.",
    "Base #1 yard destroyed, or deadline (7201 ticks).",
    80, 7203,
    'patrol bot st3 + e3; matches briefing',
    '',
    'pbox >=3 -> weap -> fact in east region', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'hard', C, 'lh-defense-tech-second-base-arena', '160x60', 'fit', T,
    'Base #1 spawns NORTH (y=22) or SOUTH (y=38) by seed; MCV per spawn',
    'Two patrol bands at BOTH latitudes (north + south); sentinel',
    "Commander, this is a secure-expand operation with seed-rotated staging and an added phase. Base #1 spawns NORTH (y=22) or SOUTH (y=38) by seed — Construction Yard, barracks, power, refinery, harvester, defenders, and a spare MCV at each. Two Soviet patrol bands hit at both latitudes. The east target region is centred at (130,30) radius 8. To win, build THREE pillboxes at base #1, have a war factory standing, deploy the MCV inside the east region, AND stand up a second refinery (proc) for total of two — all before tick 7200, with base #1 alive. Within about 80 turns.",
    "3+ pbox -> weap -> fact in east region -> 2+ proc total, all within 7200 ticks AND base#1 yard alive.",
    "Base #1 yard destroyed, or deadline (7201 ticks).",
    80, 7203,
    'patrol bot st3 + e3 on both sides; matches briefing',
    '',
    'pbox >=3 -> weap -> fact in east region -> proc >=2', 'then-strict',
    'order-spelled-out, per-phase-prescription')

# == 5. lh-econ-army-victory — econ -> army -> victory
P = 'lh-econ-army-victory'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, set_stance, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Construction Yard @(10,18) + Refinery + Power Plant + Barracks + 1 Harvester + 1 mine',
    'Enemy fact @(115,30); 2x rifle st0 HoldFire @(80, 30/32) — kill targets',
    "Commander, this is a macro chain on a fixed-tech base. Your western base — Construction Yard at (10,18), refinery, power plant, barracks, one harvester on a near ore patch (~95 cr/turn income) — has no army yet. Two enemy riflemen on hold-fire wait at (80, 30-32); a persistent enemy yard sits at (115,30). The phase markers in your win condition are: economy value reaches 1500, then you own at least six units, then you accumulate at least two unit kills. To win, all three markers must trigger in order before tick 4500. Within about 50 turns.",
    "Economy value >=1500, THEN own_units >=6, THEN units_killed >=2, all within 4500 ticks.",
    "Own yard destroyed, or deadline (4501 ticks).",
    50, 4503,
    'kill targets st0 HoldFire (intentional — gentle target supply)',
    '',
    'econ 1500 -> 6 units -> 2 kills', 'then-strict',
    'order-spelled-out')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same west base + 1 harv',
    'Enemy fact + 3x e1 st0 @(80, 28-32) kill targets',
    "Commander, this is a macro chain on a fixed-tech base with a higher capital bar. Same western base — yard, refinery, power, barracks, one harvester (~95 cr/turn). Three hold-fire enemy riflemen wait at (80, 28-32); the persistent enemy yard sits at (115,30). The phase markers are: economy value reaches 2200, then you own at least six units, then at least three kills — all in order before tick 4500. Within about 50 turns.",
    "Economy value >=2200, THEN own_units >=6, THEN units_killed >=3, all within 4500 ticks.",
    "Own yard destroyed, or deadline (4501 ticks).",
    50, 4503,
    'kill targets st0 HoldFire',
    '',
    'econ 2200 -> 6 units -> 3 kills', 'then-strict',
    'order-spelled-out')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=14) or SOUTH (y=22) by seed; same composition',
    'Enemy fact + 4x e1 st0 @(80, 28-34); both seed-mirrored mines',
    "Commander, this is a macro chain with seed-rotated staging. Your western base spawns NORTH (y=14) or SOUTH (y=22) by seed — same kit at each: yard, refinery, power, barracks, one harvester. Four hold-fire enemy riflemen wait at (80, 28-34); the persistent enemy yard sits at (115,30). The phase markers are: economy value reaches 2500, then own at least six units, then at least three kills — in order before tick 5400. Within about 60 turns.",
    "Economy value >=2500, THEN own_units >=6, THEN units_killed >=3, all within 5400 ticks.",
    "Own yard destroyed, or deadline (5401 ticks).",
    60, 5403,
    'kill targets st0 HoldFire',
    '',
    'econ 2500 -> 6 units -> 3 kills', 'then-strict',
    'order-spelled-out')

# == 6. lh-multi-checkpoint-5-plus — 3/5/6-phase PERT chain
P = 'lh-multi-checkpoint-5-plus'; C = 'action'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact + tent + powr + fix + 4x 2tnk st2 @(18-20, 18-22); 2 mines',
    'Enemy fact @(115,30) undefended; sentinel fact @(125,4)',
    "Commander, this is a three-phase chain on a fixed-tech base. Your western base — Construction Yard, barracks, power, service depot, four medium tanks on Defend stance — sits at (10,18). An undefended enemy yard waits at (115,30); a sentinel enemy yard sits far north. The phase markers in your win condition are: own a refinery (proc), own a war factory (weap), and raze the enemy yard. All three must trigger in order before tick 7200. Within about 85 turns.",
    "Own proc, THEN own weap, THEN raze enemy yard, all within 7200 ticks.",
    "Own yard destroyed, or deadline (7201 ticks).",
    85, 7653,
    'enemy passive st2; intentional (clock is the teeth)',
    '',
    'proc -> weap -> raze fact', 'then-strict',
    'order-spelled-out')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same base + 4x 2tnk',
    'Enemy fact @(115,30) + 4 HoldFire defenders (3xe1 + 1xe3); sentinel',
    "Commander, this is a five-phase chain on a fixed-tech base. Same western base — yard, barracks, power, service depot, four medium tanks — at (10,18). The enemy yard at (115,30) is now defended by three hold-fire riflemen and one rocket soldier. The phase markers are: own a refinery, own a war factory, field at least two medium tanks, accumulate at least three kills, and raze the enemy yard. All five must trigger in order before tick 7200. Within about 85 turns.",
    "Own proc, THEN own weap, THEN 2+ 2tnk, THEN 3+ kills, THEN raze enemy yard, all within 7200 ticks.",
    "Own yard destroyed, or deadline (7201 ticks).",
    85, 7653,
    'enemy st0 HoldFire (kill targets); intentional',
    '',
    'proc -> weap -> 2tnk>=2 -> kills>=3 -> raze fact', 'then-strict',
    'order-spelled-out')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=14) or SOUTH (y=26) by seed; same 4 tanks per spawn',
    'Enemy fact @(115,20) + 5 HoldFire defenders mid-band; sentinel @(125,4)',
    "Commander, this is a six-phase chain with seed-rotated staging. Your western base spawns NORTH (y=14) or SOUTH (y=26) by seed — yard, barracks, power, service depot, four medium tanks per latitude. The enemy yard at (115,20) is defended by five hold-fire soldiers. The phase markers are: own a refinery, own a war factory, field at least two medium tanks, accumulate three kills, field at least FOUR medium tanks (rebuild after attrition), and raze the enemy yard. All six in order before tick 7200. Within about 85 turns.",
    "Own proc, THEN own weap, THEN 2+ 2tnk, THEN 3+ kills, THEN 4+ 2tnk, THEN raze enemy yard, all within 7200 ticks.",
    "Own yard destroyed, or deadline (7201 ticks).",
    85, 7653,
    'enemy st0 HoldFire; intentional',
    '',
    'proc -> weap -> 2tnk>=2 -> kills>=3 -> 2tnk>=4 -> raze fact', 'then-strict',
    'order-spelled-out')

# == 7. lh-opening-to-defense-to-counter — 3-phase military chain
P = 'lh-opening-to-defense-to-counter'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, set_stance, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Yard @(10,20) + tent + proc + weap + harv; STARTS UNPOWERED (no powr); 1 mine',
    '4x rifle rush @(72,20) st3 rusher bot incoming; enemy fact @(118,20) far east + sentinel @(126,36)',
    "Commander, this is a three-phase opening drill. Your western base starts UNPOWERED — Construction Yard at (10,20), barracks, refinery, war factory, harvester, but no power plant — so the barracks cannot train infantry until you build one. A four-rifle rush is bearing down from mid-lane (x=72). A far enemy yard sits at (118,20). The phase markers are: own both a power plant and a refinery, accumulate at least four kills past tick 1100, and raze the far enemy yard — all within 5000 ticks. Within about 56 turns.",
    "Own powr+proc, THEN units_killed>=4 past tick 1100, THEN raze far yard at (118,20), all within 5000 ticks.",
    "Own yard destroyed, or deadline (5001 ticks).",
    130, 11703,
    'rusher bot st3; matches briefing',
    '',
    'powr+proc -> kills>=4@T1=1100 -> raze far fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same unpowered base',
    '6x e1 + 2x e3 rush @(72, 20-22) st3; enemy fact @(118,20) + sentinel',
    "Commander, this is a three-phase opening drill at heavier weight. Same unpowered western base — yard, barracks, refinery, war factory, harvester. A heavier rush of six riflemen and two rocket soldiers closes from x=72. Far enemy yard at (118,20). Phase markers: own powr+proc, accumulate at least seven kills past tick 1300, and raze the far yard — all within 5000 ticks. Within about 56 turns.",
    "Own powr+proc, THEN units_killed>=7 past tick 1300, THEN raze far yard, all within 5000 ticks.",
    "Own yard destroyed, or deadline (5001 ticks).",
    130, 11703,
    'rusher bot st3; matches briefing',
    '',
    'powr+proc -> kills>=7@T1=1300 -> raze far fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=14) or SOUTH (y=24) by seed; pre-placed 8x e1 st2 defender squad per spawn; unpowered',
    '8x e1 + 2x e3 rush @(72, 20-22) st3 (rush is immediate on spawn_point path); enemy fact @(118,20) + sentinel @(126,36)',
    "Commander, this is a three-phase opening drill with seed-rotated staging and an immediate rush. Your unpowered western base spawns NORTH (y=14) or SOUTH (y=24) by seed with a pre-built eight-rifle defensive squad. Eight riflemen and two rocket soldiers rush immediately. The far enemy yard at (118,20) is the counter objective. Phase markers: own powr+proc, accumulate nine kills past tick 700, and raze the far yard — all within 5000 ticks. Within about 56 turns.",
    "Own powr+proc, THEN units_killed>=9 past tick 700, THEN raze far yard, all within 5000 ticks.",
    "Own yard destroyed, or deadline (5001 ticks).",
    130, 11703,
    'rusher bot st3; matches briefing',
    '',
    'powr+proc -> kills>=9@T1=700 -> raze far fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')

# == 8. lh-opening-to-tech-to-army — 4-phase macro chain
P = 'lh-opening-to-tech-to-army'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact + tent + powr + fix (no proc, no weap); 2 mines',
    'Enemy fact @(115,30) + 1xe1 + 1xe3 st2 passive defenders (bot disabled)',
    "Commander, this is a four-phase macro chain. Your western base — Construction Yard at (10,18), barracks, power plant, service depot — sits without a refinery or war factory. The far enemy yard at (115,30) has two passive defenders. Phase markers: own a refinery, own a war factory, field at least two medium tanks, and raze the enemy yard. All four in order before tick 8999. Within about 100 turns.",
    "Own proc, THEN own weap, THEN 2tnk>=2, THEN raze enemy yard, all within 8999 ticks.",
    "Own yard destroyed, or deadline (9000 ticks).",
    100, 9003,
    'enemy passive st2 (bot disabled); intentional',
    '',
    'proc -> weap -> 2tnk>=2 -> raze fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same base',
    'Enemy fact + 1xe1 hunt-bot drip @(110,30); sentinel',
    "Commander, this is a four-phase macro chain with light pressure. Same western base — yard, barracks, power, service depot — no refinery or war factory. The far enemy yard at (115,30) is defended by one rifleman; a light hunt-bot drip pressures your base. Phase markers: own a refinery, own a war factory, field at least FOUR medium tanks, and raze the enemy yard. All four in order before tick 7200. Within about 80 turns.",
    "Own proc, THEN own weap, THEN 2tnk>=4, THEN raze enemy yard, all within 7200 ticks.",
    "Own yard destroyed, or deadline (7201 ticks).",
    80, 7203,
    'hunt-bot st3 light drip; matches briefing',
    '',
    'proc -> weap -> 2tnk>=4 -> raze fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=14) or SOUTH (y=26) by seed; 1 starter rifleman per spawn st2; bot disabled',
    'Enemy fact @(115,30) + 2xe1 st2 passive; far sentinel @(120,5)',
    "Commander, this is a four-phase macro chain with seed-rotated staging. Your western base spawns NORTH (y=14) or SOUTH (y=26) by seed — yard, barracks, power, service depot, one starter rifleman per spawn. The far enemy yard at (115,30) is defended by two passive riflemen. Phase markers: own a refinery, own a war factory, field at least FIVE medium tanks, and raze the enemy yard. All four in order before tick 9300. Within about 110 turns.",
    "Own proc, THEN own weap, THEN 2tnk>=5, THEN raze enemy yard, all within 9300 ticks.",
    "Own yard destroyed, or deadline (9301 ticks).",
    110, 9903,
    'enemy passive st2 (bot disabled); intentional',
    '',
    'proc -> weap -> 2tnk>=5 -> raze fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')

# == 9. lh-progression-stage-locked — 5-stage PERT chain
P = 'lh-progression-stage-locked'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, set_stance, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact ONLY @(10,18) + 1 harv @(20,18) parked at mine + 4x 2tnk st0 @(18-20, 22-24); 2 mines',
    '3x e1 st0 HoldFire @(90, 28-32) kill targets; 2x sentinel facts far east',
    "Commander, this is a five-stage PERT-style chain. Your western base is bare — Construction Yard only at (10,18) — with a harvester parked beside the ore patch and four medium tanks on hold-fire at (18-20, 22-24). Three hold-fire enemy riflemen wait at (90, 28-32) as kill targets; two sentinel enemy yards sit far east. Stage markers: own a power plant, own a refinery, reach economy value 3500, own a war factory, and accumulate at least two UNIT kills. Razing any enemy building forfeits. All five in order before tick 6000. Within about 70 turns.",
    "Own powr, THEN own proc, THEN econ>=3500, THEN own weap, THEN unit kills>=2, all within 6000 ticks. Razing any enemy building forfeits.",
    "Own yard destroyed, ANY enemy building destroyed, or deadline (6001 ticks).",
    70, 6303,
    'kill targets st0 HoldFire; intentional',
    '',
    'powr -> proc -> econ 3500 -> weap -> unit kills>=2', 'then-strict',
    'order-spelled-out, outcome-table')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same bare base + 4x 2tnk',
    '3x e1 + 1x e3 st0 @(90, 28-32); sentinels',
    "Commander, this is a five-stage PERT-style chain with a tighter capital bar. Same bare western base — yard, parked harvester, four hold-fire medium tanks. Four hold-fire enemy soldiers (three riflemen and one rocket soldier) wait at (90, 28-32). Two sentinel enemy yards far east. Stage markers: own a power plant, own a refinery, reach economy value 3000, own a war factory, accumulate at least three UNIT kills. Razing any enemy building forfeits. All five in order before tick 6000. Within about 70 turns.",
    "Own powr, THEN own proc, THEN econ>=3000, THEN own weap, THEN unit kills>=3, all within 6000 ticks. Razing any enemy building forfeits.",
    "Own yard destroyed, ANY enemy building destroyed, or deadline (6001 ticks).",
    70, 6303,
    'kill targets st0 HoldFire; intentional',
    '',
    'powr -> proc -> econ 3000 -> weap -> unit kills>=3', 'then-strict',
    'order-spelled-out, outcome-table')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=14) or SOUTH (y=26) by seed; bare fact + 1 harv + 4 tanks per spawn',
    '5x st0 mid-band kill targets @(90, 18-22); sentinels @(120,8) @(125,4)',
    "Commander, this is a five-stage PERT chain with seed-rotated staging. Your western base spawns NORTH (y=14) or SOUTH (y=26) by seed — Construction Yard, parked harvester, four hold-fire medium tanks. Five hold-fire enemy soldiers wait at (90, 18-22). Stage markers: own a power plant, own a refinery, reach economy value 3000, own a war factory, accumulate three UNIT kills. Razing any enemy building forfeits. All five in order before tick 6000. Within about 70 turns.",
    "Own powr, THEN own proc, THEN econ>=3000, THEN own weap, THEN unit kills>=3, all within 6000 ticks. Razing any enemy building forfeits.",
    "Own yard destroyed, ANY enemy building destroyed, or deadline (6001 ticks).",
    70, 6303,
    'kill targets st0 HoldFire; intentional',
    '',
    'powr -> proc -> econ 3000 -> weap -> unit kills>=3', 'then-strict',
    'order-spelled-out, outcome-table')

# == 10. lh-recovery-after-mid-game-loss — disaster recovery
P = 'lh-recovery-after-mid-game-loss'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact @(8,18) + powr + weap WEST (safe) + proc @(40,18) FORWARD + harv + 4x 3tnk (2 inside blast, 2 SOUTH safe @ y=26); 1 mine',
    '5x e1 st1 garrison @(82, 16-22) + far sentinel @(122,20); turtle bot',
    "Commander, this is a disaster-recovery operation. Your Soviet command — Construction Yard at (8,18) plus power plant and war factory on the safe west — runs a forward Ore Refinery at (40,18) with a harvester nearby and four heavy tanks staged around it. A five-rifle eastern garrison at (82, 16-22) defends the enemy yard. At tick ~1500 a scheduled sabotage blast wipes everything in radius 6 of (40,18): the refinery, the harvester, and the two forward tanks; the other two tanks (south at y=26) and the west base survive. To win, kill at least five enemy units AND own at least one refinery, after tick 1600, all within 5400 ticks. Within about 60 turns.",
    "After tick 1600: kills>=5 AND own a refinery, within 5400 ticks.",
    "Own yard destroyed, or deadline (5401 ticks).",
    60, 5403,
    'turtle bot — garrison stance:1 ReturnFire, intentional; scheduled_events at tick 1500',
    '',
    'kills>=5 & proc>=1 after t=1600', 'all_of-scheduled',
    'outcome-table')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same base layout',
    '6x e1 st1 compact garrison @(82-84, 17-21) + sentinel; turtle bot; HEAVIER blast (r=8 wipes 3 of 4 tanks)',
    "Commander, this is a disaster-recovery operation with a heavier blast. Same Soviet command — west fact, power, war factory, forward refinery, harvester, four heavy tanks. The eastern garrison is six riflemen at (82-84, 17-21). At tick ~1500 a heavier sabotage blast (radius 8 around (40,18)) wipes the refinery, harvester, AND THREE of four tanks — only one survivor remains. To win, kill at least six enemy units AND own at least one refinery, after tick 1600, all within 4600 ticks. Within about 52 turns.",
    "After tick 1600: kills>=6 AND own a refinery, within 4600 ticks.",
    "Own yard destroyed, or deadline (4601 ticks).",
    52, 4683,
    'turtle bot st1 + scheduled_events tick 1500',
    '',
    'kills>=6 & proc>=1 after t=1600', 'all_of-scheduled',
    'outcome-table')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=12) or SOUTH (y=28) by seed; full base + column duplicated at each',
    'Same compact garrison @(82-84, 18-22) + sentinel; turtle; disaster region declared at BOTH latitudes',
    "Commander, this is a disaster-recovery operation with seed-rotated staging. Your Soviet command spawns NORTH (y≈12) or SOUTH (y≈28) by seed — full base, forward refinery, harvester, four heavy tanks, lone WEST survivor at x=20. The eastern garrison is six riflemen mid-band at (82-84, 18-22). At tick 1500 a sabotage blast (radius 8) fires at BOTH latitudes — only the active one removes assets. To win, kill at least six enemy units AND own at least one refinery, after tick 1600, all within 4600 ticks. A memorised rebuild cell mis-places on the flipped seed. Within about 52 turns.",
    "After tick 1600: kills>=6 AND own a refinery, within 4600 ticks.",
    "Own yard destroyed, or deadline (4601 ticks).",
    52, 4683,
    'turtle bot st1 + scheduled_events at both latitudes',
    '',
    'kills>=6 & proc>=1 after t=1600 (rebuild after sabotage)', 'all_of-scheduled',
    'outcome-table')

# == 11. lh-scout-react-counter — observation-driven 3-phase
P = 'lh-scout-react-counter'; C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact + powr + 1 jeep st1 + 3x 2tnk st2 @(20, 18-20)',
    'Enemy fact @(122,18) + 4x e1 st0; isolated enemy barracks @(60,36); sentinel @(124,4)',
    "Commander, this is an observation-driven three-phase drill. Your western base has a jeep scout on Return-Fire and three medium tanks on Defend at (20, 18-20). The enemy main is an undefended yard at (122,18) ringed by four hold-fire riflemen at (118,18); a SEPARATE enemy barracks outpost sits at (60,36), well off the main assault axis. Phase markers: discover at least two enemy buildings, accumulate three kills, raze the enemy yard. All three in order before tick 6300. Within about 75 turns.",
    "Discover >=2 enemy buildings, THEN kills>=3, THEN raze enemy yard, within 6300 ticks AND own yard alive AND >=1 own unit.",
    "Own yard destroyed, all own units dead, or deadline (6301 ticks).",
    75, 6753,
    'enemy st0 HoldFire kill cluster; intentional',
    '',
    'discover>=2 -> kills>=3 -> raze fact', 'then-strict',
    'order-spelled-out')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=10) or SOUTH (y=30) by seed; jeep + 3x 2tnk per spawn',
    'NE enemy fact @(122,5) + 4xe1 st0; SE @(122,38) + 4xe3 st0; shared outpost barracks @(60,20)',
    "Commander, this is an observation-driven three-phase drill with seed-rotated staging. Your western base spawns NORTH (y=10) or SOUTH (y=30) by seed — jeep on Return-Fire and three medium tanks on Defend per spawn. TWO enemy yards hide at the NE corner (with riflemen) and SE corner (with rocket soldiers); a shared enemy barracks outpost sits mid-map at (60,20). Phase markers: discover at least two enemy buildings, three kills, raze a yard. All three in order before tick 6300. Within about 75 turns.",
    "Discover >=2 enemy buildings, THEN kills>=3, THEN raze enemy yard, within 6300 ticks AND own yard alive AND >=1 own unit.",
    "Own yard destroyed, all own units dead, or deadline (6301 ticks).",
    75, 6753,
    'enemy st0 HoldFire kill clusters; intentional',
    '',
    'discover>=2 -> kills>=3 -> raze fact', 'then-strict',
    'order-spelled-out')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=10) or SOUTH (y=30) by seed',
    'THREE enemy yards (NE 122,5 + MID 122,22 + SE 122,38) each with 4-unit cluster; outpost barracks @(60,20); attrition cap',
    "Commander, this is an observation-driven three-phase drill with three hidden compositions and an attrition cap. Your western base spawns NORTH (y=10) or SOUTH (y=30) by seed — jeep on Return-Fire and three medium tanks on Defend. THREE enemy yards hide far east (NE, MID, SE), each with a defending cluster of about four units; a shared enemy barracks outpost sits at (60,20). Phase markers: discover at least two enemy buildings, three kills, raze a yard — AND lose no more than three of your own. All within 6300 ticks. Within about 75 turns.",
    "Discover >=2, THEN kills>=3, THEN raze enemy yard, AND units_lost<=3, within 6300 ticks AND own yard alive AND >=1 own unit.",
    "Own yard destroyed, all own units dead, attrition>3 lost, or deadline (6301 ticks).",
    75, 6753,
    'enemy st0 HoldFire kill clusters; intentional',
    '',
    'discover>=2 -> kills>=3 -> raze fact + loss cap 3', 'then-strict',
    'order-spelled-out')

# == 12. lh-tech-pivot-attack — late tech-pivot
P = 'lh-tech-pivot-attack'; C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact + powr + tent + weap + fix + 1 jeep st1 @(20,20)',
    'Enemy fact @(122,18) + 4x 2tnk st0 cluster @(118,18); sentinel @(124,4)',
    "Commander, this is a late-game pivot drill on a fixed-tech base. Your western base — Construction Yard, power plant, barracks, war factory, service depot, one scout jeep at (20,20) — has the production substrate ready but no army. The far enemy yard at (122,18) is defended by four hold-fire medium tanks. Phase markers: the jeep reaches the eastern scout band (radius 12 around (120,18)), you produce at least four rocket soldiers, and you raze the enemy yard. All three in order before tick 7200. Within about 85 turns.",
    "Jeep in (120,18) r=12, THEN 4+ rocket soldiers, THEN raze enemy yard, within 7200 ticks.",
    "Own yard destroyed, or deadline (7201 ticks).",
    85, 7653,
    'enemy st0 HoldFire cluster; intentional',
    '',
    'scout (jeep east) -> 4xe3 -> raze fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=10) or SOUTH (y=30) by seed; full tech + jeep per spawn',
    'NE enemy fact @(122,5) + 6xe3 st0 cluster; SE fact @(122,38) + 4x 2tnk st0 cluster',
    "Commander, this is a late-game pivot drill with hidden compositions. Your western base spawns NORTH (y=10) or SOUTH (y=30) by seed — full tech (yard, power, barracks, war factory, service depot) and one scout jeep. Two enemy yards hide at the east corners — one defended by rocket soldiers, one by medium tanks — with the composition flipping per seed. Phase markers: the jeep reaches either eastern scout band, you commit either four rocket soldiers OR (a gun turret + two medium tanks), and you raze a yard. All three in order before tick 7200. Within about 85 turns.",
    "Jeep in NE or SE band, THEN (4xe3 OR gun+2x 2tnk), THEN raze a yard, within 7200 ticks.",
    "Own yard destroyed, or deadline (7201 ticks).",
    85, 7653,
    'enemy st0 HoldFire clusters; intentional',
    '',
    'scout (jeep east band) -> matching counter -> raze fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=10) or SOUTH (y=30) by seed; full tech + jeep per spawn',
    'Same NE 6xe3 + SE 4x2tnk clusters; tighter clock 6300',
    "Commander, this is a late-game pivot drill with seed-rotated staging and a tighter clock. Your western base spawns NORTH (y=10) or SOUTH (y=30) by seed — full tech and one scout jeep. Two enemy yards at the east corners, each with a hold-fire defending cluster, composition varying by seed. Phase markers: the jeep reaches either eastern scout band, you commit four rocket soldiers OR a gun turret plus two medium tanks, and you raze a yard. All three in order before tick 6300. Within about 75 turns.",
    "Jeep in NE or SE band, THEN matching counter (4xe3 OR gun+2x 2tnk), THEN raze a yard, within 6300 ticks.",
    "Own yard destroyed, or deadline (6301 ticks).",
    75, 6753,
    'enemy st0 HoldFire clusters; intentional',
    '',
    'scout (jeep east band) -> matching counter -> raze fact', 'then-strict',
    'order-spelled-out, per-phase-prescription')

# == 13. lh-tech-rush-vs-army-rush — capex vs opex under fast rush
P = 'lh-tech-rush-vs-army-rush'; C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact + proc + powr + tent + weap @(10-16, 20-24) — tech up, no starter units; cash $1800',
    '5x e1 rusher band @(48,20) st3; far sentinel fact @(120,20)',
    "Commander, this is a commit-under-pressure drill. Your western base has full tech already standing — Construction Yard, refinery, power plant, barracks, war factory at (10-16, 20-24) — and 1800 credits to spend. No starting units. A fast five-rifle rush is bearing down from x=48 (about thirty cells out). To win, accumulate at least four kills AND keep the Construction Yard standing, before tick 4800. Within about 54 turns.",
    "Kills>=4 AND yard alive, within 4800 ticks.",
    "Own yard destroyed, or deadline (4801 ticks).",
    60, 5403,
    'rusher bot st3; matches briefing',
    '',
    'kills>=4 & yard alive (flat all_of)', 'all_of-terminal',
    'outcome-table')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same tech-up base; cash $1800',
    '6x e1 + 2x e3 rusher band @(44, 20) st3 (closer); sentinel',
    "Commander, this is a commit-under-pressure drill at heavier weight. Your western base has full tech up — Construction Yard, refinery, power, barracks, war factory — and 1800 credits. No starting units. Six riflemen and two rocket soldiers rush from x=44 (about thirty cells out). To win, accumulate at least six kills AND keep the yard standing, before tick 4800. Within about 54 turns.",
    "Kills>=6 AND yard alive, within 4800 ticks.",
    "Own yard destroyed, or deadline (4801 ticks).",
    60, 5403,
    'rusher bot st3; matches briefing',
    '',
    'kills>=6 & yard alive (flat all_of)', 'all_of-terminal',
    'outcome-table')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Base spawns NORTH (y=12) or SOUTH (y=28) by seed; full tech + 1xe1 st0 witness per spawn; $1800',
    'Two rusher bands — 5xe1 + 1xe3 at y=12 AND y=28 — both always place; sentinel @(120,20)',
    "Commander, this is a commit-under-pressure drill with seed-rotated staging. Your western base spawns NORTH (y=12) or SOUTH (y=28) by seed — full tech standing, one hold-fire rifle witness per spawn, and 1800 credits. Two fast rushes (five rifles plus one rocket soldier per latitude) close from x=48. To win, accumulate at least seven kills AND keep the yard standing, before tick 4800. A memorised army-placement cell fails on the flipped seed. Within about 54 turns.",
    "Kills>=7 AND yard alive, within 4800 ticks.",
    "Own yard destroyed, or deadline (4801 ticks).",
    60, 5403,
    'rusher bot st3; matches briefing',
    '',
    'kills>=7 & yard alive (flat all_of)', 'all_of-terminal',
    'outcome-table')

# == 14. longhorizon-opening-to-assault — 4-phase all_of terminal chain
P = 'longhorizon-opening-to-assault'; C = 'reasoning'
T = 'build, place_building, move_units, attack_unit, deploy, stop_units'
add(P, 'easy', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'fact @(10,18) + powr + tent (no proc); cash 5000',
    'Enemy fact @(115,30) + proc @(115,34) + 2x e1 + 1x e3 st2 defenders',
    "Commander, this is a long-horizon chain across the full map. Your western base — Construction Yard, power plant, barracks — sits at (10,18) with 5000 credits. The eastern enemy site has a yard, refinery and three passive defenders at (115,30-34). To win, by deadline you must have explored at least 35% of the map, own a refinery, have at least four combat units, and have at least one unit inside radius 8 of (115,32). Within about 121 turns.",
    "Explored >=35% AND own proc AND own_units>=4 AND at least one unit at (115,32) r=8, all within 10803 ticks.",
    "All own units dead, or deadline (after tick 10700).",
    120, 10803,
    'enemy passive st2; intentional (no bot)',
    '',
    'explore 35% + proc + 4 units + reach (115,32) r=8', 'all_of-terminal',
    'order-spelled-out')
add(P, 'medium', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same base + cash 5000',
    'Same enemy site',
    "Commander, this is a long-horizon chain at a tighter target. Your western base — Construction Yard, power plant, barracks — at (10,18) with 5000 credits. The eastern enemy site has a yard, refinery and three passive defenders at (115,30-34). To win, by deadline you must have explored at least 45% of the map, own a refinery, have at least five combat units, and have at least one unit inside radius 7 of (115,32). Within about 141 turns.",
    "Explored >=45% AND own proc AND own_units>=5 AND at least one unit at (115,32) r=7, all within 12603 ticks.",
    "All own units dead, or deadline (after tick 12500).",
    140, 12603,
    'enemy passive st2; intentional',
    '',
    'explore 45% + proc + 5 units + reach (115,32) r=7', 'all_of-terminal',
    'clean')
add(P, 'hard', C, 'rush-hour-arena', '128x40', 'large-trivial', T,
    'Same base + cash 4500',
    'Same enemy site',
    "Commander, this is a long-horizon chain with an attrition cap. Your western base — Construction Yard, power plant, barracks — at (10,18) with 4500 credits. The eastern enemy site has a yard, refinery and three passive defenders. To win, by deadline you must have explored at least 55% of the map, discovered at least one enemy building, own a refinery, have at least six combat units, have at least one unit inside radius 6 of (115,32), AND have lost no more than four of your own. Within about 161 turns.",
    "Explored >=55% AND >=1 enemy building discovered AND own proc AND own_units>=6 AND at least one unit at (115,32) r=6 AND units_lost<=4, all within 14403 ticks.",
    "All own units dead, or deadline (after tick 14300).",
    160, 14403,
    'enemy passive st2; intentional',
    '',
    'explore 55% + discover>=1 + proc + 6 units + reach (115,32) r=6 + loss cap 4', 'all_of-terminal',
    'order-spelled-out')


# == Emit CSV ==
def main():
    fieldnames = [
        'pack', 'level', 'capability', 'map_name', 'map_size',
        'map_fit', 'tools', 'agent_force', 'enemy_force',
        'enemy_posture', 'posture_issue', 'briefing_RA',
        'win_condition', 'lose_condition', 'max_turns', 'tick_budget',
        'phase_chain', 'chain_idiom', 'leak_flags',
    ]
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in R:
            w.writerow({k: row.get(k, '') for k in fieldnames})
    print(f"wrote {len(R)} rows to {OUT}")


if __name__ == '__main__':
    main()
