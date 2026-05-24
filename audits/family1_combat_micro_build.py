"""Builds audits/family1_combat_micro.csv — Family-1 (Combat micro) manual audit.

One row per (pack, level). Each briefing is SELF-CONTAINED — the model
sees one level at a time, so every briefing fully describes the mission,
the forces given, and the objective from scratch. No "same as before"
or "the same X" references.

Structure per briefing: mission framing → what is given → target/objective.
Red-Alert-specific terms (stance, kiting, pillbox, etc.) are explained
inline so non-RA readers can follow.

The map_fit column flags scenarios where the map is too large for the
actual decision under test (turning the scenario into search-and-destroy
instead of testing the advertised capability).
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent / 'family1_combat_micro.csv'
R = []

def add(pack, level, cap, map_name, map_size, map_fit, tools, agent, enemy,
        briefing, win, lose, max_turns, tick_budget,
        posture='', posture_issue=''):
    R.append(dict(
        pack=pack, level=level, capability=cap, map_name=map_name,
        map_size=map_size, map_fit=map_fit, tools=tools,
        agent_force=agent, enemy_force=enemy,
        briefing_RA=briefing, win_condition=win, lose_condition=lose,
        max_turns=max_turns, tick_budget=tick_budget,
        enemy_posture=posture, posture_issue=posture_issue,
    ))

# 1. action-multiunit-coordination — [EDITED 2026-05-23] map rush-hour-arena 128x40 → arena 48x40
P='action-multiunit-coordination'; C='action'; M='arena 48x40'; SZ='48x40'
T='move_units, attack_unit, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '3× medium tank @(4,6) + 3× light tank @(6,10), all on west edge, all on HOLD-FIRE',
    '1× Construction Yard @(44,4) NE; 1× Ore Refinery @(44,34) SE',
    "Commander, this is a parallel strike mission. You are given a tank column on the west edge: 3 medium tanks and 3 light tanks. Two enemy buildings sit on the far east of the map — a Construction Yard at the north-east corner and an Ore Refinery at the south-east. Split your force into two groups and hit both at the same time, because a one-at-a-time push will not beat the clock.",
    'Have ≥2 units inside a radius-6 ring around BOTH targets within 2200 ticks.',
    'All units destroyed, or deadline (2201 ticks).',
    30, 2700)
add(P,'medium',C,M,SZ,'fit',T,
    '3× medium tank @(4,6) + 3× light tank @(6,10) + 2× jeep @(4,24), all HOLD-FIRE',
    '1× Construction Yard @(44,4) NE, 1× Ore Refinery @(44,34) SE, 1× Power Plant @(12,36) SW, 2× rifle infantry @(24,6) + 2× @(24,32) mid pickets',
    "Commander, this is a three-front parallel strike mission. You are given 3 medium tanks, 3 light tanks, and 2 jeeps on the west side. Three enemy buildings must be reached at the same time: a Construction Yard in the north-east, an Ore Refinery in the south-east, and a Power Plant in the south-west. Small rifle-infantry pickets sit in the middle of the map. Split your force into three groups, drive in parallel, and hold losses to no more than two.",
    'Have ≥2 units in a radius-6 ring around ALL THREE targets within 2800 ticks, ≤2 own lost.',
    'All units dead, >2 lost, or deadline (2801 ticks).',
    36, 3333)
add(P,'hard',C,M,SZ,'fit',T,
    'Same force as medium (3× 2tnk + 3× 1tnk + 2× jeep, HOLD-FIRE)',
    'Same three buildings + same pickets; coords HIDDEN — only direction labels given',
    "Commander, this is a three-front parallel strike mission with no coordinates given. You are given 3 medium tanks, 3 light tanks, and 2 jeeps on the west side. Three enemy buildings must be reached at the same time, each labelled on your map by direction only: an enemy Construction Yard in the north-east, an enemy Power Plant in the south-west, and an enemy Ore Refinery in the south-east. Push columns toward each direction to scout for them; the instant a unit spots a target you will be interrupted and shown its exact location. Hold losses to no more than two.",
    'Have ≥2 units in each of the three labelled regions within 2800 ticks, ≤2 lost.',
    'All dead, >2 lost, or deadline (2801 ticks).',
    36, 3333)

# 2. action-sequenced-execution — [EDITED 2026-05-23] maps shrunk:
# easy rush-hour 128x40 → sequenced-easy 48x32; medium → sequenced-medium 48x40;
# hard scout-arena 176x80 → sequenced-hard 80x48
P='action-sequenced-execution'; C='action'; M='arena 48x32'; SZ='48x32'
T='move_units, attack_unit, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '2× medium tank @(4,8) + 2× light tank @(6,12), HOLD-FIRE',
    'Three landmark buildings (no defenders): Power Plant @(16,24), Ore Refinery @(32,6), Construction Yard @(44,28)',
    "Commander, this is an ordered patrol mission. You are given 2 medium tanks and 2 light tanks on the west edge. A three-stop route through enemy territory has already been plotted: an enemy Power Plant in the mid-south first, then an Ore Refinery in the north, then a Construction Yard in the far south-east. Drive your column through the route in order and do not let the tanks sit idle between stops — issue the next move the instant a unit reaches the previous stop.",
    'Visit waypoints in order (16,24) → (32,6) → (44,28), radius-6 reach, within 2400 ticks.',
    'All units dead, or deadline (2401 ticks).',
    30, 2700)
add(P,'medium',C,'arena 48x40','48x40','fit',T,
    'North: 2× 2tnk @(4,8) + 1× 1tnk @(6,10). South: 2× 1tnk @(4,32) + 1× jeep @(6,30). HOLD-FIRE.',
    '6 landmark buildings @(16/32/44, 8/32) + 2× rifle infantry @(28,20) mid screen',
    "Commander, this is a two-column ordered patrol. You are given one north column (2 medium tanks + 1 light tank) and one south column (2 light tanks + 1 jeep). Each column has its own ordered route: the north column runs north→south→north, the south column runs south→north→south. Each waypoint is an enemy building. Drive both columns in parallel, do not halt between stops, and hold losses to one or fewer.",
    'Both chains (N: (16,8)→(32,32)→(44,8); S: (16,32)→(32,8)→(44,32)) completed, within 3000 ticks, ≤1 loss.',
    'All dead, >1 lost, or deadline (3001 ticks).',
    36, 3333)
add(P,'hard',C,'arena 80x48','80x48','fit',T,
    'Two seed-spawned columns: 2× 2tnk + 2× 1tnk + 2× jeep each, mirrored far N (sp0 @(10,10)) / far S (sp1 @(10,38))',
    '8 landmark buildings forming two serpentine chains + 2× rifle infantry @(42,24) + 2× rocket infantry @(56,24)',
    "Commander, this is a long-range ordered patrol on a large 80x48 map. You are given two columns of 6 units each (2 medium tanks + 2 light tanks + 2 jeeps), one deployed far north and one far south. Each column must run a four-stop ordered route across the map, with stops labelled north, centre, east, and far corner. Coordinates are NOT given — each waypoint is an enemy building hidden in fog; scout toward the labelled direction and you will be interrupted the instant a unit spots one, revealing its exact location. Drive both routes in parallel and hold losses to one or fewer.",
    'Both 4-waypoint chains (A & B) completed in order, within 6000 ticks, ≤1 loss.',
    'All dead, >1 lost, or deadline (6001 ticks).',
    70, 6303)

# 3. combat-attack-from-behind-fog — [EDITED 2026-05-23] map 128x40 → 64x40;
# line x=50 → x=24; fact (100,20) → (48,20); survival bar ≥2 → ≥3 to make
# brute attack_move decisively LOSE (previously squeaked through with lost=2)
P='combat-attack-from-behind-fog'; C='reasoning'; M='arena 64x40 (2 obs)'; SZ='64x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) on Return-Fire stance',
    '3× rocket infantry @(24,19-21); target Construction Yard @(48,20); sentinel fact @(60,4)',
    "Commander, this is a flanking strike against an undefended enemy Construction Yard. You are given 4 medium tanks on the west edge. A thin anti-tank line — 3 rocket infantry — covers the central corridor at x=24. Driving head-on through the corridor will burn the clock without reaching the yard at (48,20) behind the line. A clear NORTH fog lane runs above the central walls; take it, swing east past the line’s longitude, then turn south to destroy the yard. Keep at least 3 tanks alive within about 60 turns.",
    'Destroy rear Construction Yard (radius-6 of (48,20)) with ≥3 tanks alive, within 5400 ticks.',
    '<3 tanks left, or deadline (5401 ticks).',
    72, 6483)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) on Return-Fire',
    '2 pillboxes (stationary defensive turrets) + 5 rocket infantry at x=24; target Construction Yard @(48,20)',
    "Commander, this is a flanking strike against an undefended enemy Construction Yard. You are given 4 medium tanks on the west edge. A heavy line covers the central corridor at x=24 — 2 pillboxes (stationary defensive turrets) flanking 5 rocket infantry. A head-on push cooks the column in overlapping kill envelopes. Open fog lanes run above (north) and below (south) the central walls. Take one of them, drive east past the line’s longitude, then turn in to destroy the yard at (48,20). Keep at least 3 tanks alive within about 50 turns.",
    'Destroy rear Construction Yard with ≥3 tanks alive, within 4500 ticks.',
    '<3 tanks left, or deadline (4501 ticks).',
    60, 5403)
add(P,'hard',C,M,SZ,'fit',T,
    '4× medium tank seed-rotated N (sp0 @4,14-17) or S (sp1 @4,23-26), Return-Fire',
    '3 pillboxes + 8 rocket infantry filling x=24 y=15-25; target Construction Yard @(48,20)',
    "Commander, this is a flanking strike against an undefended enemy Construction Yard. You are given 4 medium tanks staged in either the north corridor (y=14..17) or the south corridor (y=23..26) depending on seed. A dense wall — 3 pillboxes interleaved with 7 rocket infantry — covers the central corridor at x=24. A head-on charge dies in the crossfire. Loop wide via your nearest fog lane (north y=2..5 if you staged north, south y=35..37 if you staged south), drive east past the line, then turn in to destroy the enemy yard at (48,20). Keep at least 3 tanks alive within about 50 turns.",
    'Destroy rear Construction Yard with ≥3 tanks alive, within 4500 ticks.',
    '<3 tanks left, or deadline (4501 ticks).',
    60, 5403)

# 4. combat-bait-counter-attack — [EDITED 2026-05-23] map 112x40 → 56x40;
# fact (80,20)→(40,20); guards x=76 → x=36; forward spawn (18,…)→(12,…); sentinel (108,4)→(52,4)
P='combat-bait-counter-attack'; C='reasoning'; M='arena 56x40 (3 obs)'; SZ='56x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(4,12) main + 1× jeep @(4,28) decoy',
    'Construction Yard @(40,20) + 3× rocket infantry guards @(36,18-22) (guard bot, lunge ~16); sentinel fact @(52,4)',
    "Commander, this is a bait-and-strike mission on a guarded enemy Construction Yard. You are given 4 medium tanks staged on the north flank (y=12) and 1 jeep staged south (y=28). The yard at (40,20) is guarded by 3 rocket infantry that hold their post but will lunge at any enemy within ~16 cells and then snap back past a leash of ~18. Send the jeep on a deep south-east vector to pull the guards off post, then drive the tank column onto the now-undefended yard. Destroy the yard with at most 3 losses, within about 60 turns.",
    'Destroy the enemy Construction Yard, ≤3 own losses, within 3600 ticks.',
    'All dead, >3 lost, or deadline (3601 ticks).',
    62, 5583)
add(P,'medium',C,M,SZ,'fit',T,
    '5× medium tank @(4,12) + 1× jeep @(4,28)',
    'Construction Yard @(40,20) + 5× rocket infantry in L-cover (west @(36,18-20) + south @(36-44,24))',
    "Commander, this is a bait-and-strike mission on a guarded enemy Construction Yard. You are given 5 medium tanks staged on the north flank (y=12) and 1 jeep staged south (y=28). The yard at (40,20) is guarded by 5 rocket infantry forming an L across the west and south faces. The guards hold post but lunge at any enemy within ~16 cells. Send the jeep on a deep south-east vector to pull the south arc off post, then swing the tanks around the now-vacated north flank and onto the yard. Destroy the yard with at most 2 losses, within about 60 turns.",
    'Destroy the Construction Yard, ≤2 own losses, within 3600 ticks.',
    'All dead, >2 lost, or deadline.',
    62, 5583)
add(P,'hard',C,M,SZ,'fit',T,
    'Two seed-spawned staging columns: 5× 2tnk + 1× jeep each, far-west (sp0 @4,…) or forward (sp1 @12,…)',
    'Construction Yard @(40,20) + 5× rocket-infantry L-cover (same as medium)',
    "Commander, this is a bait-and-strike mission on a guarded enemy Construction Yard. You are given 5 medium tanks staged on the north flank (y=12) plus 1 jeep on the south (y=28). On this seed you may stage far west (x=4) or well forward (x=12) — read the actual position from the observation. The yard at (40,20) is guarded by 5 rocket infantry in an L across its west and south faces. Guards hold post but lunge within ~16 cells. Send the jeep on a deep south-east vector to pull the south arc off post, then swing the tanks around the now-vacated north flank and onto the yard. Destroy the yard with at most 2 losses, within about 60 turns.",
    'Destroy the Construction Yard, ≤2 losses, within 3600 ticks.',
    'All dead, >2 lost, or deadline.',
    62, 5583)

# 5. combat-divide-and-conquer — [EDITED 2026-05-23] map 128x40 → 64x40;
# clusters x=60 → x=28; sentinel (120,…) → (60,…); hard wall obstacles
# rescaled to span x=10..55
P='combat-divide-and-conquer'; C='reasoning'; M='arena 64x40'; SZ='64x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) + Construction Yard @(2,20)',
    '2× rocket infantry @(28,14-16) NORTH + 2× @(28,24-26) SOUTH stance:3; sentinel fact @(60,4)',
    "Commander, this is a divide-and-conquer engagement. You are given 4 medium tanks on the west edge and a Construction Yard at our rear. Two enemy rocket-infantry teams sit bracketed at the centre — one north at y=14-16, one south at y=24-26. A straight east push along y=20 puts the lead tank under fire from both teams at once and the column bleeds. Flank wide north (or south) to break line-of-sight on the far team, destroy the near team in isolation, then pivot to the other flank and finish the second team. Kill at least 4 enemies, keep ≥3 tanks alive AND hold the Construction Yard, within about 40 turns.",
    '≥4 kills, ≥3 alive, hold Construction Yard, within 3500 ticks.',
    '<3 alive, Construction Yard lost, or deadline (3501 ticks).',
    60, 5403)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) + Construction Yard @(2,20)',
    '2 teams @ x=28: NORTH 3× e3 + 1× 1tnk @ y=14-16, SOUTH 3× e3 + 1× 1tnk @ y=24-26 (stance:3)',
    "Commander, this is a divide-and-conquer engagement. You are given 4 medium tanks on the west edge and a Construction Yard at our rear. Two reinforced enemy teams sit bracketed at the centre — each with 3 rocket infantry plus a light tank, one team north at y=14-16 and one south at y=24-26. A head-on midpoint push eats fire from both teams at once. Flank wide, destroy one team in isolation, then pivot to the other side and finish the second. Kill all 8 enemies, keep ≥3 tanks alive AND hold the Construction Yard, within about 40 turns.",
    '≥8 kills, ≥3 alive, hold Construction Yard, within 3500 ticks.',
    '<3 alive, Construction Yard lost, or deadline.',
    60, 5403)
add(P,'hard',C,M,SZ,'fit',T,
    '4× medium tank seed-spawned NORTH (sp0 @4,4-7) or SOUTH (sp1 @4,33-36) + Construction Yard',
    'Three corridors split by walls (y=9-15, y=25-30); 3 teams @ x=28 (NE @y=5, MID @y=20, SW @y=35) each 2× e3 + 1× 1tnk; sentinel @(60,20)',
    "Commander, this is a divide-and-conquer engagement against three enemy teams. You are given 4 medium tanks staged in either the north corridor (y=4..7) or the south corridor (y=33..36) depending on seed, plus a Construction Yard at our rear in the same corridor. The arena is split into three lanes — north (y=2..8), middle (y=17..23), south (y=32..37) — by impassable walls. Three enemy teams sit at x=28: one in each corridor (NE, MID, SW), 3 units per team. A straight east push through the middle corridor's choke gets cooked. Engage your near corridor's team first, traverse to the opposite corridor via the open west sliver, then engage the remaining teams. Kill all 9 enemies, keep ≥3 tanks alive AND hold the Construction Yard, within about 40 turns.",
    '≥9 kills, ≥3 alive, hold Construction Yard, within 3500 ticks.',
    '<3 alive, no Construction Yard, or deadline.',
    60, 5403)

# 6. combat-flanking-attack — [EDITED 2026-05-23] map 128x40 → 32x40,
# spawn (6,20)→(4,20), engagement (58-60)→(22-24), far fact (120,20)→(28,20)
P='combat-flanking-attack'; C='action'; M='arena 32x40 (2 obs)'; SZ='32x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) on the west edge',
    '5× rocket infantry @(24,18-22) + 5× rifle infantry shield @(22,18-22); water funnel x=8..20, NORTH detour y=2..5 OPEN, SOUTH sealed',
    "Commander, this is a flank-attack mission. You are given 4 medium tanks on the west edge. The centre lane is a water-walled funnel with an anti-tank line waiting inside it — 5 rocket infantry behind a 5-man rifle shield. Driving through the funnel will destroy the column. A clear NORTH detour lane runs above the water; take it to swing around and hit the line from the side.",
    'At least 3 kills, ≥3 tanks alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline (4501).',
    51, 4593)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) on the west edge',
    '7× rocket infantry @(24,17-23) + 5× rifle infantry shield @(22,18-22); water funnel x=8..20, BOTH north and south detours open',
    "Commander, this is a flank-attack mission. You are given 4 medium tanks on the west edge. A heavy enemy line sits across the centre — 7 rocket infantry backed by 5 rifle infantry — inside a water-walled funnel. The funnel is a kill zone; do not drive through it. Open detour lanes run above (north) and below (south) the water; take one of them and hit the line from the side.",
    'At least 4 kills, ≥3 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline.',
    51, 4593)
add(P,'hard',C,M,SZ,'fit',T,
    '8× medium tank split N (sp0 @4,14-17) / S (sp1 @4,23-26)',
    '9× rocket infantry front rank @x=24 (y=15-25) + 4× back rank @x=26 + 7× rifle shield @x=22; water funnel x=8..20',
    "Commander, this is a flank-attack mission with two attacking groups. You are given 8 medium tanks split between far north and far south. A full-strength enemy wall sits across the centre — 13 rocket infantry in two rows backed by 7 rifle infantry — inside a water-walled funnel. The funnel is a kill zone. Both groups must come around the open ground above and below the water and hit the wall from the sides at the same time.",
    'At least 5 kills, ≥3 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline.',
    51, 4593)

# 7. combat-focus-fire-priority — [EDITED 2026-05-23] maps shrunk
# rush-hour 128x40 → easy 40x40 / medium 32x40 / hard 32x40.
# Enemy line x=70-73 → x=20-23; medium/hard staging x=60 → x=6/x=10.
# Hard attrition cap loosened 0→1 (lead tank takes one shot on
# approach before focus-fire drops first e3 on tighter geometry).
P='combat-focus-fire-priority'; C='action'; M='arena 40x40'; SZ='40x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-21) Return-Fire',
    '4× rifle infantry @(20,17/19/21/23) shield + 3× rocket infantry @(21,18/20/22) rear-rank (st2); sentinel @(36,20)',
    "Commander, this is a focus-fire target-priority drill. You are given 4 medium tanks on the west edge. The enemy is a mixed squad at x=20-21 — a 4-man rifle-infantry shield wall with 3 anti-tank rocket soldiers behind it. The rocket soldiers deal far more damage to your tanks than the rifles. Concentrate ALL four tanks' fire on one rocket soldier at a time (4-vs-1 kills it in 1-2 turns), then the next rocket, then the third, then mop up the rifles. Kill all 7 enemies with zero own losses, within about 30 turns.",
    '≥7 kills, 0 losses, within 2700 ticks.',
    'Any loss, all dead, or deadline (2701).',
    30, 2700)
add(P,'medium',C,'arena 32x40','32x40','fit',T,
    '4× medium tank @(6,18-21) staged just outside Dragon range',
    '8× rifle infantry @(20,15-25) wide shield + 2× rocket infantry @(21,18/22) rear-rank; sentinel @(28,4)',
    "Commander, this is a focus-fire target-priority drill. You are given 4 medium tanks staged just outside the enemy's rocket range. The enemy is a centre squad — an 8-man rifle-infantry shield wall at x=20 backed by 2 anti-tank rocket soldiers at x=21. The rocket soldiers are the priority threat. Concentrate ALL four tanks' fire on one rocket at a time, then the second, then mop up the rifles. Kill all 10 enemies with zero own losses, within about 30 turns.",
    '≥10 kills, 0 losses, within 2700 ticks.',
    'Any loss, deadline.',
    30, 2700)
add(P,'hard',C,'arena 32x40','32x40','fit',T,
    '4× medium tank seed-spawned NORTH (sp0 @10,15-18) or SOUTH (sp1 @10,22-25)',
    '12× rifle infantry @(20-22, 17/19/21/23) compact 3-column shield + 2× rocket infantry @(23,18/22) rear-rank; sentinel @(28,20)',
    "Commander, this is a focus-fire target-priority drill with seed-rotated staging. You are given 4 medium tanks staged in the north (y=15..18) or south (y=22..25) corridor at x=10, depending on seed. The enemy is a compact centre squad — a 12-man rifle-infantry shield wall spanning y=17..23 at x=20-22, backed by 2 anti-tank rocket soldiers at x=23. The rocket soldiers are the priority threat. Concentrate ALL four tanks' fire on one rocket at a time, then the second, then mop up the rifles. Kill all 14 enemies with at most 1 loss, within about 30 turns.",
    '≥14 kills, ≤1 loss, within 2700 ticks.',
    '>1 lost, all dead, or deadline.',
    30, 2700)

# 8. combat-formation-tank-wedge
P='combat-formation-tank-wedge'; C='action'; M='rush-hour-arena'; SZ='128x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'large-trivial',T,
    '5× medium tank @(6,18-22)',
    '5× rocket infantry firing into the corridor from above + 1× light tank blocking the road',
    "Commander, this is a wedge push through hostile fire. You are given 5 medium tanks on the west edge. The corridor east is under fire from 5 rocket infantry above and is blocked on-axis by a light tank. Keep the wedge tight, focus fire on the road blocker, and reach the rally point at (80,20) with at least 4 tanks intact.",
    'At least 4 tanks within radius-6 of (80,20), ≥4 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline.',
    75, 6753)
add(P,'medium',C,M,SZ,'large-trivial',T,
    '5× medium tank @(6,18-22)',
    '3× rocket infantry firing from above + 3× from below + 1× light tank blocking the road',
    "Commander, this is a wedge push through bracketing fire. You are given 5 medium tanks on the west edge. The corridor east is under fire from rocket infantry on both flanks — 3 above and 3 below — and is blocked on-axis by a light tank. Hold formation, kill the blocker, push through the bracket, and rally at (80,20).",
    'At least 3 kills, ≥3 in the rally region, ≥3 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline.',
    75, 6753)
add(P,'hard',C,M,SZ,'large-trivial',T,
    '10× medium tank in two groups on the far north and far south flanks',
    '8× rocket infantry firing from above and below + 2× light tanks blocking the road',
    "Commander, this is a wedge push with two converging columns. You are given 10 medium tanks split between far north and far south. The corridor east is bracketed by 8 rocket infantry (above and below) and blocked on-axis by 2 light tanks. Each wedge punches through one blocker; both columns meet at the rally point (80,20) with at least 3 tanks intact.",
    'At least 4 kills, ≥3 in rally, ≥3 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline.',
    75, 6753)

# 9. combat-harass-aggro-commit — [EDITED 2026-05-23] map rush-hour-arena
# 128x40 → tailored arena 56x40. Defender x=75 → x=36; harv cluster
# x=78/82 → x=38/42; proc x=80 → x=40; sentinel (120,20) → (52,4).
P='combat-harass-aggro-commit'; C='action'; M='arena 56x40'; SZ='56x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(4,18-22) on the west edge, Return-Fire',
    '2× harvester (unarmed ore-collector trucks) @(38,19)/(42,21) + 1× heavy tank defender @(36,20) + Ore Refinery @(40,20); sentinel fact @(52,4)',
    "Commander, this is an aggressive raid against an enemy economy. You are given 4 medium tanks on the west edge (x=4). An enemy Ore Refinery sits at (40,20) with 2 harvesters working the patch and a single heavy tank guard at (36,20) — the heavy out-trades a medium tank one-on-one. Focus your fire on the guard first, then sweep the harvesters. Score 3 kills with at least 1 tank surviving, within about 50 turns.",
    'At least 3 kills, ≥1 of your units alive, within 4500 ticks.',
    'All dead, or deadline (4501 ticks).',
    51, 4593)
add(P,'medium',C,M,SZ,'fit',T,
    '3× medium tank @(4,18-22) on the west edge, Return-Fire',
    '3× harvester @(38,19)/(42,21)/(42,18) + 1× heavy tank defender @(36,20) + Ore Refinery @(40,20)',
    "Commander, this is an aggressive raid against an enemy economy. You are given 3 medium tanks on the west edge. The Ore Refinery at (40,20) is guarded by one heavy tank at (36,20) and worked by 3 harvesters. Three of your tanks shooting the guard together still beat it. Focus on the guard first, then sweep the harvesters. Four kills, one tank surviving, within about 50 turns.",
    'At least 4 kills, ≥1 alive, within 4500 ticks.',
    'All dead, or deadline (4501 ticks).',
    51, 4593)
add(P,'hard',C,M,SZ,'fit',T,
    '3× medium tank seed-spawned NORTH (sp0 @4,10-14) or SOUTH (sp1 @4,26-30) corridor',
    '4× harvester around proc @(40,20) + 2× heavy tank defenders @(36,16) N + @(36,24) S + Ore Refinery',
    "Commander, this is an aggressive raid against an enemy economy. You are given 3 medium tanks staged in the north (y=10..14) or south (y=26..30) corridor depending on seed. The Ore Refinery at (40,20) is guarded by two heavy tanks — one north at (36,16), one south at (36,24) — and worked by 4 harvesters. Focus on ONE guard at a time, then the other, then sweep the harvesters. Splitting fire across both heavies lets them grind you down. Six kills, one raider surviving, within about 50 turns.",
    'At least 6 kills, ≥1 alive, within 4500 ticks.',
    'All dead, or deadline (4501 ticks).',
    51, 4593)

# 10. combat-harass-balanced-hit-and-run — [EDITED 2026-05-23] map
# rush-hour-arena 128x40 → tailored arena 96x40 (3/4 the old width;
# the guard-aggro/leash distances + the long jeep transit needed for
# COMMIT to die before reaching the 3-kill bar are load-bearing and
# can't compress further). Positions preserved from the OLD layout
# (jeeps x=6, workers x=72..92, guard (82,5), rally x=50).
P='combat-harass-balanced-hit-and-run'; C='reasoning'; M='arena 96x40'; SZ='96x40'
T='move_units, attack_unit, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '2× jeep @(6,10) Return-Fire',
    '5× rifle-infantry workers HoldFire @(72/76/80/84/88, 10) + 1× heavy tank guard @(82,5) (guard bot, aggro 16, leash 18); sentinel fact @(92,20)',
    "Commander, this is a pulsed worker-harass mission. You are given 2 jeeps on the west edge at y=10. Five unarmed worker infantry stretch along y=10 from x=72 to x=88. One enemy heavy tank stands one row north at (82,5) — your jeep machine guns cannot scratch its armour, and its cannon one-shots a jeep in two turns. It lunges at anything within about 16 cells of its post and snaps back past about 18. Strike a worker, immediately retreat west of x=50, let it snap back, then re-engage. Three kills, zero jeeps lost, within about 60 turns.",
    'At least 3 kills, ZERO losses, within 5400 ticks.',
    'Any loss, or deadline (5401 ticks).',
    62, 5583)
add(P,'medium',C,M,SZ,'fit',T,
    '2× jeep @(6,10) Return-Fire',
    '6× rifle-infantry workers HoldFire @(72/76/80/84/88/92, 10) + 1× heavy tank guard @(82,5)',
    "Commander, this is a pulsed worker-harass mission with a bigger target cluster. You are given 2 jeeps on the west edge at y=10. Six worker infantry stretch along y=10 from x=72 to x=92. The enemy heavy tank sits one row north at (82,5); your machine guns cannot dent it, its cannon one-shots a jeep, and it lunges at anything within about 16 cells of its post. Pulse: strike one worker, retreat west of x=50, let it snap back, re-engage. Three kills, zero jeeps lost, within about 60 turns.",
    'At least 3 kills, ZERO losses, within 5400 ticks.',
    'Any loss, or deadline (5401 ticks).',
    62, 5583)
add(P,'hard',C,M,SZ,'fit',T,
    '4× jeep seed-spawned NORTH (sp0 @6,10) or SOUTH (sp1 @6,30) lane',
    '12× rifle-infantry workers (6 per lane) + 2× heavy tank guards @(82,5) N + @(82,35) S',
    "Commander, this is a pulsed worker-harass mission with twin cluster lanes. You are given 4 jeeps staged in the north (y=10) or south (y=30) lane depending on seed. Each lane mirrors the same setup: six unarmed worker infantry along x=72..92 on the lane latitude, and a heavy tank one row off the lane (north post (82,5); south post (82,35)). Pulse: strike a worker, retreat west of x=50, let the heavy snap back, re-engage. Three kills total, zero jeeps lost, within about 60 turns.",
    'At least 3 kills total, ZERO losses, within 5400 ticks.',
    'Any loss, or deadline (5401 ticks).',
    62, 5583)

# 11. combat-heli-flank
P='combat-heli-flank'; C='action'; M='rush-hour-arena'; SZ='128x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '1× Construction Yard @(4,20) + 2× helicopter @(30,19-21)',
    'A 23-cell vertical wall of pillboxes (stationary defensive turrets) at x=50 + 3× rifle infantry cluster behind it',
    "Commander, this is an air-mobility strike. You are given 2 helicopters sitting at (30, 19..21) and a Construction Yard at (4,20) on the west side. A solid wall of pillboxes (stationary defensive turrets, impassable to ground units) runs vertically at x=50 from y=10..30 and cuts the map in half — no ground unit can pass through it inside the deadline. Three enemy riflemen sit behind the wall at (60, 19..21). Fly the helicopters over the wall and gun all three down. Keep the Construction Yard intact, within about 20 turns.",
    'At least 3 kills AND your Construction Yard is still standing, within 1800 ticks.',
    'Construction Yard destroyed, or deadline (1801).',
    21, 1893)
add(P,'medium',C,M,SZ,'fit',T,
    '1× Construction Yard @(4,20) + 2× helicopter @(30,19-21)',
    '23-cell pillbox wall at x=50 + 5× rifle infantry cluster behind it',
    "Commander, this is an air-mobility strike. You are given 2 helicopters sitting at (30, 19..21) and a Construction Yard at (4,20) on the west side. A solid wall of pillboxes (stationary defensive turrets, impassable to ground units) runs vertically at x=50 from y=10..30 and cuts the map in half — no ground unit can pass through it inside the deadline. Five enemy riflemen are clustered behind the wall at (60, 18..22). Fly the two helicopters over the wall and clear the whole line. Keep the Construction Yard standing, within about 24 turns.",
    'At least 5 kills, hold Construction Yard, within 2100 ticks.',
    'Construction Yard destroyed, or deadline.',
    24, 2163)
add(P,'hard',C,M,SZ,'fit',T,
    '2× helicopter staged in north (y=11..13) OR south (y=27..29) by seed @x=30 + 1× Construction Yard @(4,20)',
    '23-cell pillbox wall at x=50 + 5× rifle infantry cluster behind it',
    "Commander, this is an air-mobility strike with seed-rotated staging. You are given 2 helicopters staged at x=30 in EITHER the north corridor (y=11..13) OR the south corridor (y=27..29) — read your actual starting band from the observation — plus a Construction Yard at (4,20). A solid wall of pillboxes (stationary defensive turrets, impassable to ground units) runs vertically at x=50 from y=10..30 and cuts the map in half. Five enemy riflemen wait behind the wall at (60, 18..22). Fly the helicopters over the wall from your starting band, gun all five down, and keep the Construction Yard standing, within about 20 turns.",
    'At least 5 kills, hold Construction Yard, within 1800 ticks.',
    'Construction Yard destroyed, or deadline.',
    21, 1893)

# 12. combat-hold-chokepoint
P='combat-hold-chokepoint'; C='action'; M='chokepoint-arena 96x40'; SZ='96x40'
T='observe, move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '3× medium tank in the narrow pass @(44,19-21) + Construction Yard @(8,20)',
    '12× light tanks in a grid at x=80-86 + enemy Construction Yard',
    "Commander, this is a chokepoint defence on a 96x40 arena. You are given 3 medium tanks parked at the mouth of a four-cell-wide corridor (around x=44, y=18..21) and a Construction Yard far west at (8,20). The corridor is the only opening through an impassable water wall that splits the map at x=45..51. Twelve enemy light tanks stage deep in the eastern lobe and hunt you through the corridor. Stay anchored at the mouth and concentrate ALL three tanks' fire on the lead attacker — the water forces the swarm to a four-abreast trickle, so a focused front kills faster than the queue can attrit you. Abandoning the mouth (charging east or retreating west into the open) lets the full force surround you. Nine kills, at least two tanks still alive at the mouth, Construction Yard intact, within about 60 turns.",
    'At least 9 kills, ≥2 tanks alive, hold Construction Yard, ≥2 tanks still inside the chokepoint, within 5400 ticks.',
    'Fewer than 2 alive, or deadline (5401).',
    75, 6753)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank in the chokepoint + Construction Yard',
    '14× light tanks at x=80-86 + enemy Construction Yard',
    "Commander, this is a chokepoint defence on a 96x40 arena. You are given 4 medium tanks at the mouth of a three-cell-wide corridor (around x=43..44, y=19..21) and a Construction Yard far west at (8,20). The corridor is the only opening through an impassable water wall that splits the map at x=45..51. Fourteen enemy light tanks stage in the eastern lobe and hunt you through the corridor. Anchor at the mouth and concentrate fire on the lead attacker — the water forces the swarm to a three-abreast trickle. Abandoning the mouth (charging east or retreating west into the open) lets the full force surround you. Eleven kills, at least three tanks anchored at the mouth, Construction Yard intact, within about 60 turns.",
    'At least 11 kills, ≥3 alive, ≥3 in chokepoint, hold Construction Yard, within 5400 ticks.',
    'Fewer than 2 alive, or deadline.',
    75, 6753)
add(P,'hard',C,M,SZ,'fit',T,
    '4× medium tank at the 2-wide corridor mouth (x=43..44, y=19..20) + Construction Yard @(8,20)',
    '14× light tanks staged in NORTH (y=10..14) OR SOUTH (y=26..30) cluster by seed + enemy Construction Yard @(92,20)',
    "Commander, this is a chokepoint defence on a 96x40 arena, the tightest tier. You are given 4 medium tanks at the mouth of a two-cell-wide corridor (around x=43..44, y=19..20) and a Construction Yard far west at (8,20). The corridor is the only opening through an impassable water wall that splits the map at x=45..51. Fourteen enemy light tanks stage in the eastern lobe — from a north (y=10..14) OR south (y=26..30) cluster depending on seed — and hunt you through the corridor. Anchor at the mouth and concentrate fire on the lead attacker; the water forces the swarm to a two-abreast queue regardless of which side they staged from. Abandoning the mouth lets the full force surround you. Nine kills, at least three tanks anchored at the mouth, Construction Yard intact, within about 60 turns.",
    'At least 9 kills, ≥3 alive, ≥3 in chokepoint, hold Construction Yard, within 5400 ticks.',
    'Fewer than 2 alive, or deadline.',
    75, 6753)

# 13. combat-kite-and-pull — [EDITED 2026-05-23] map tailored 128x40
# → 96x40 (3/4 the old width). Kiter staging x=20 preserved; heavy
# at x=80 (easy/med) / x=70 (hard) preserved. Fact x=124 → x=92.
# The 18-cell west retreat space is load-bearing — a tighter arena
# strands the kiter on the cordon before the kite cycle completes.
# Engine-verified 2026-05-23: bot_type:hunt + stance:2 DOES drive
# movement (explicit Attack order overrides stance per CLAUDE.md).
P='combat-kite-and-pull'; C='action'; M='arena 96x40'; SZ='96x40'
T='observe, move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '1× medium tank @(20,10) on north corridor, Return-Fire',
    '1× wounded heavy tank @(80,20) (35% HP, hunt bot); sentinel fact @(92,20)',
    "Commander, this is a kiting engagement — fire from range, then move back before the enemy closes. You are given 1 medium tank at (20,10) on the north corridor. The enemy is a wounded heavy tank at (80,20) — its cannon hits harder than yours and it out-trades you head-on. When the heavy closes inside seven cells, move your tank back along the lane; otherwise shoot from range. Kill the heavy with your tank still alive, within about 50 turns.",
    'Kill the heavy tank, ≥1 of your tanks alive, within 4500 ticks.',
    'All dead, or deadline (4501 ticks).',
    51, 4593)
add(P,'medium',C,M,SZ,'fit',T,
    '2× medium tank @(20,9)/(20,11) paired on north corridor, Return-Fire',
    '1× wounded heavy tank @(80,20) (40% HP, hunt bot)',
    "Commander, this is a kiting engagement against a wounded heavy tank. You are given 2 medium tanks in tight formation on the north corridor at (20,9) and (20,11). One enemy heavy tank hunts from (80,20); it out-trades you head-on. Kite as a pair — back off whenever the heavy enters seven cells, snipe from outside — and kill the heavy with BOTH tanks alive, within about 50 turns.",
    'Kill the heavy, ≥2 alive, within 4500 ticks.',
    'Fewer than 2 alive, or deadline (4501 ticks).',
    51, 4593)
add(P,'hard',C,M,SZ,'fit',T,
    '3× medium tank seed-spawned NORTH (sp0 @28-30,9-11) or SOUTH (sp1 @28-30,29-31) corridor',
    '1× healthy heavy tank @(70,20) (70% HP, hunt bot)',
    "Commander, this is a kiting engagement against a healthy heavy tank. You are given 3 medium tanks in the north (y=10) or south (y=30) corridor depending on seed. One enemy heavy tank hunts from the mid latitude — its cannon hits harder than yours. The kite-and-pull cycle is the same — retreat past seven cells, attack from range — but the clock tightens to about 40 turns. All three tanks must survive.",
    'Kill the heavy, ≥3 alive, within 3600 ticks.',
    'Fewer than 3 alive, or deadline (3601 ticks).',
    41, 3693)

# 14. combat-kite-jeep-vs-tank — [EDITED 2026-05-23] map tailored
# 128x40 → 96x40 (3/4 the old width). Raider staging @(28-30,9-11)
# preserved; heavy at (70,20) easy / (80,20) med+hard preserved.
# Fact (124,20) → (92,20). Engine-verified: bot_type:hunt + stance:2
# DOES drive movement (explicit Attack overrides stance).
P='combat-kite-jeep-vs-tank'; C='action'; M='arena 96x40'; SZ='96x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '3× medium tank @(28-30,9-11) on north corridor, Return-Fire',
    '1× heavy tank @(70,20) actively hunting (hunt bot); sentinel fact @(92,20)',
    "Commander, this is a kiting engagement against a hunting enemy heavy tank. You are given 3 medium tank raiders at (28,9), (30,10), and (28,11). The enemy heavy at (70,20) is actively closing; its cannon hits harder than yours and can one-shot a raider at close range. Kite: when it closes within seven cells, move away along the lane, then attack from outside its lethal close-range window. All three raiders must survive, within about 50 turns.",
    'Kill the heavy, ≥3 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline (4501 ticks).',
    51, 4593)
add(P,'medium',C,M,SZ,'fit',T,
    '3× medium tank @(28-30,9-11), Return-Fire',
    '1× heavy tank further east @(80,20) actively hunting (hunt bot)',
    "Commander, this is a kiting engagement against a hunting enemy heavy tank, deeper east. You are given 3 medium tank raiders at (28,9), (30,10), and (28,11). The enemy heavy at (80,20) is actively closing; its cannon out-trades you head-on. Standing still trades one raider for the kill. Kite: when the heavy enters seven cells, move away along the lane; otherwise fire from range. Kill the heavy without losing one of three raiders, within about 50 turns.",
    'Kill the heavy, ≥3 alive, within 4500 ticks.',
    'Fewer than 3 alive, or deadline (4501 ticks).',
    51, 4593)
add(P,'hard',C,M,SZ,'fit',T,
    '3× medium tank seed-spawned NORTH (sp0 @28-30,9-11) or SOUTH (sp1 @28-30,29-31) corridor',
    '1× heavy tank @(80,20) actively hunting (hunt bot)',
    "Commander, this is a kiting engagement against a hunting enemy heavy tank. You are given 3 medium tank raiders in the NORTH (y=10) or SOUTH (y=30) corridor depending on seed. The enemy heavy at (80,20) actively closes on whichever corridor you spawn in. The cannon out-trades raider weapons at close range. Kite: when the heavy enters seven cells, move your raiders away along the lane; otherwise fire from range. All three raiders must survive, within about 40 turns.",
    'Kill the heavy, ≥3 alive, within 3600 ticks.',
    'Fewer than 3 alive, or deadline (3601 ticks).',
    41, 3693)

# 15. combat-naval-shore-strike
P='combat-naval-shore-strike'; C='action'; M='naval-arena 64x40'; SZ='64x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '1× destroyer (warship) @(15,20)',
    '1× rifle infantry on the shore @(18,20) + far Construction Yard',
    "Commander, this is a coastal bombardment. You are given 1 destroyer (a naval warship that floats on water and cannot move onto land) sitting in the western water channel at (15,20). A lone enemy rifleman is dug in on the shore at (18,20), three cells east. The destroyer's naval gun reaches across the shoreline. Move into range if needed and shell the rifleman down, within about 30 turns.",
    'At least 1 kill, within 2699 ticks.',
    'Destroyer lost, or deadline (2700).',
    32, 2883)
add(P,'medium',C,M,SZ,'fit',T,
    '2× destroyer',
    '3× rifle infantry on the shore',
    "Commander, this is a coastal bombardment. You are given 2 destroyers (naval warships that float on water and cannot move onto land) in the western water channel at (15,18) and (15,22). Three enemy rifle infantry are dug in on the shore at x=18 (y=18, 20, 22). The destroyers' naval guns reach across the shoreline. Hold position offshore and shell all three riflemen down, within about 30 turns.",
    'At least 3 kills, within 2699 ticks.',
    'All destroyers lost, or deadline.',
    32, 2883)
add(P,'hard',C,M,SZ,'fit',T,
    '2× destroyer',
    '5-man shore garrison including rocket infantry that can hit ships',
    "Commander, this is a coastal bombardment against a hardened shore. You are given 2 destroyers (naval warships that float on water and cannot move onto land) in the western water channel at (15,18) and (15,22). An enemy shore garrison at x=18 (y=17..23) is 4 riflemen plus 1 anti-tank rocket soldier — the rocket soldier carries an anti-armour launcher that CAN damage your destroyers, so an unlucky volley sinks a ship. Shell the rocket soldier FIRST before he ranges you, then clean up the four riflemen. Five kills with at least one destroyer still afloat, within about 30 turns.",
    'At least 5 kills, within 2699 ticks.',
    'All destroyers lost, or deadline.',
    32, 2883)

# 16. combat-pincer-coordination — [EDITED 2026-05-23] map shrunk
# 144x40 → 72x40; cluster x=50 → x=22; sentinel (120,4) → (60,4);
# obstacles trimmed to two distant marker dots (the original 4-rock
# bracket squeezed converging tanks into pocket mouths on the smaller
# canvas). Medium/hard cluster reduced from 2× enemy 2tnks to 1× to
# keep the synchronised pincer inside the loss cap after the shrink.
P='combat-pincer-coordination'; C='action'; M='arena 72x40 (2 obs)'; SZ='72x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '3× medium tank @(5,8) north + 3× medium tank @(5,32) south',
    'Central enemy cluster @(22,20): 4× rifle infantry + 2× rocket infantry + far Construction Yard sentinel @(60,4)',
    "Commander, this is a pincer attack. You are given two squads of 3 medium tanks on opposite flanks — one north at (5,8), one south at (5,32). The enemy cluster at (22,20) is 4 rifle infantry and 2 rocket infantry. Both squads must hit the cluster at the same time — if one arrives first, the cluster focuses fire on it and destroys it. Hold losses to two or fewer.",
    'At least 6 kills, ≥4 units in the (22,20) region together, ≤2 losses, within 4500 ticks.',
    'More than 2 lost, all dead, or deadline.',
    64, 5763)
add(P,'medium',C,M,SZ,'fit',T,
    '3× medium tank @(5,8) north + 3× medium tank @(5,32) south',
    'Central enemy cluster @(22,20): 4× rifle infantry + 2× rocket infantry + 1× medium tank',
    "Commander, this is a pincer attack against a reinforced cluster. You are given two squads of 3 medium tanks on opposite flanks — one north at (5,8), one south at (5,32). The enemy cluster at (22,20) is 4 rifle infantry, 2 rocket infantry, and 1 medium tank. Both squads must hit the cluster simultaneously, or the cluster will destroy whichever column arrives first. Hold losses to two or fewer.",
    'At least 6 kills, ≥4 in the (22,20) region, ≤2 losses, within 4500 ticks.',
    'More than 2 lost, all dead, or deadline.',
    64, 5763)
add(P,'hard',C,M,SZ,'fit',T,
    'Two seed-driven layouts: sp0 @(5,8)/(5,32) or sp1 @(3,5)/(3,35), each 3× medium tank per squad',
    'Central reinforced cluster (4× rifle + 2× rocket + 1× medium tank) at (22,20)',
    "Commander, this is a pincer attack with seed-varied staging. You are given two squads of 3 medium tanks each on opposing latitudes along the west edge — exact cells vary by seed (north plus south staging). The enemy cluster at (22,20) is 4 rifle infantry, 2 rocket infantry, and 1 medium tank. All prongs must hit the cluster at the same moment. Hold losses to two or fewer.",
    'At least 6 kills, ≥4 in the (22,20) region, ≤2 losses, within 4500 ticks.',
    'More than 2 lost, all dead, or deadline.',
    64, 5763)

# 17. combat-prevent-retreat — [EDITED 2026-05-23] posture defect
# fixed (Option B: briefing rewritten as a FIXED encirclement instead
# of flee — the stance:2 cluster never moves, so the "infantry will
# run east" premise was false). Map shrunk 112x40 → 56x40: cluster
# x=60 → x=30, cut-off disc (85,20) → (43,20), sentinel (105,4) →
# (52,4). Hard cluster rebalanced to 2× e3 + 6× e1 (was 3+5) to keep
# the focus-fire intended play inside the loss cap on the tighter
# canvas.
P='combat-prevent-retreat'; C='action'; M='arena 56x40 (4 obs)'; SZ='56x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(6,18-21)',
    '6-unit alternating rifle/rocket infantry block @(30,17-22) on Defend stance + far sentinel @(52,4)',
    "Commander, this is a fixed-encirclement mission. You are given 4 medium tanks on the west edge. An enemy infantry block holds the centre at (30, y=17..22) — 6 units alternating rifle and rocket infantry — on Defend stance (auto-fires when you wander into range, but never moves). To pin and finish them you must close the back door first: slip one tank around the flank to the cut-off disc at (43,20), then engage the centre with the main body.",
    'At least 6 kills AND ≥1 of your units at (43,20), ≥3 alive, within 3500 ticks.',
    'Fewer than 3 alive, or deadline.',
    60, 5403)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank @(6,18-21)',
    '7-unit alternating rifle/rocket infantry block @(30,16-22) on Defend stance',
    "Commander, this is a fixed-encirclement mission. You are given 4 medium tanks on the west edge. An enemy infantry block holds the centre at (30, y=16..22) — 7 units alternating rifle and rocket — on Defend stance and will not move. Send one tank around the flank to the cut-off disc at (43,20), then engage with the main body.",
    'At least 7 kills + cut-off (43,20) held + ≥3 alive, within 3500 ticks.',
    'Fewer than 3 alive, or deadline.',
    60, 5403)
add(P,'hard',C,M,SZ,'fit',T,
    '4× medium tank staged NORTH (y=12-15) or SOUTH (y=25-28) by seed',
    '8-unit infantry block at (30, y=16..23): 2× e3 + 6× e1, on Defend stance',
    "Commander, this is a fixed-encirclement mission with seed-rotated staging. You are given 4 medium tanks at the west edge — staged in either the north corridor (y=12..15) or the south corridor (y=25..28) depending on seed. An enemy infantry block holds (30, y=16..23) — 8 units (2 rocket soldiers + 6 rifles) on Defend stance and will not move. Send one tank around the appropriate flank to the cut-off disc at (43,20), then engage with the main body.",
    'At least 8 kills + cut-off (43,20) held + ≥3 alive, within 3500 ticks.',
    'Fewer than 3 alive, or deadline.',
    60, 5403)

# 18. combat-protect-vip-escort
P='combat-protect-vip-escort'; C='action'; M='arena 144x40 (6 obs)'; SZ='144x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '1× harvester VIP @(6,20) (unarmed, 70% HP) + 4× medium tank escort @(6, 18/19/21/22)',
    'Ambush 1 @ x=49: 1× heavy tank @y=20 + 4× rifle infantry @y=17/18/22/23. Ambush 2 @ x=85: 3× rifle infantry @y=18/20/22.',
    "Commander, this is a VIP escort mission across a long 144x40 lane. You are given 1 harvester (an unarmed ore-collector truck — your VIP, cannot fight back, starting at 70% HP) at (6,20), plus 4 medium tank escorts straddling it at (6, 18/19/21/22). The harvester is the ONLY one of its type on the map; if it dies the mission is lost. Two ambushes block the route east: at x=49 sits a guard force of 1 enemy heavy tank (hits harder than your mediums; will lunge at and run down a lone harvester) plus 4 riflemen, and at x=85 sits a smaller clump of 3 riflemen. Move the escort tanks AHEAD to break each ambush in turn while holding the VIP back at staging; only after the route is cleared, send the harvester across to the extraction disc around (130,20). Within about 60 turns.",
    'VIP harvester inside radius-6 of (130,20), ≥1 harvester alive, within 5400 ticks.',
    'VIP dead, or deadline (5401).',
    61, 5493)
add(P,'medium',C,M,SZ,'fit',T,
    '1× harvester VIP @(6,20) (unarmed, 70% HP) + 4× medium tank escort @(6, 18/19/21/22)',
    'Ambush 1 @ x=49: 1× heavy tank + 6× rifle infantry spanning y=16..24. Ambush 2 @ x=85: 3× rifle infantry @y=18/20/22.',
    "Commander, this is a VIP escort mission across a long 144x40 lane. You are given 1 harvester (an unarmed ore-collector truck — your VIP, cannot fight back, starting at 70% HP) at (6,20), plus 4 medium tank escorts straddling it at (6, 18/19/21/22). The harvester is the ONLY one of its type on the map; if it dies the mission is lost. Two ambushes block the route east: at x=49 sits a tall guard line of 1 enemy heavy tank (hits harder than your mediums; will lunge at a lone harvester) plus 6 riflemen spread across y=16..24, and at x=85 sits a smaller clump of 3 riflemen. Move the escort tanks AHEAD to break each ambush in turn while holding the VIP back at staging; only after the route is cleared, send the harvester across to the extraction disc around (130,20). Within about 60 turns.",
    'VIP at extraction, alive, within 5400 ticks.',
    'VIP dead, or deadline.',
    61, 5493)
add(P,'hard',C,M,SZ,'fit',T,
    '1× harvester VIP + 4× medium tank escort staged NORTH (y≈16) OR SOUTH (y≈24) by seed',
    'Ambush 1 @ x=49: 2× heavy tanks (one per lane) at health:55 + 5× rifle infantry spanning y=16..24. Ambush 2 @ x=85: 5× rifle infantry spanning y=16..24.',
    "Commander, this is a VIP escort mission across a long 144x40 lane, with seed-rotated staging. You are given 1 harvester (an unarmed ore-collector truck — your VIP, cannot fight back, starting at 70% HP) plus 4 medium tank escorts, staged together at either the NORTH row (around y=16) or the SOUTH row (around y=24) depending on seed — read your actual starting row from the observation. The harvester is the ONLY one of its type on the map; if it dies the mission is lost. The route holds two enemy heavy tanks at x=49 — one aligned to each staging lane, each at 55% HP, each able to lunge at and destroy an unescorted harvester — backed by a wide rifle line spanning y=16..24, plus a second rifle clump at x=85 spanning the same band. Move the escort tanks AHEAD to break the interceptors while holding the VIP back at staging; only after the route is clear, send the harvester across to the extraction disc around (130,20). Within about 60 turns.",
    'VIP at extraction, alive, within 5400 ticks.',
    'VIP dead, or deadline.',
    61, 5493)

# 19. combat-retreat-after-engagement — [EDITED 2026-05-23] map
# shrunk 112x40 → 56x40 (was tagged `large-trivial`). e3 firing line
# x=60 → x=30; 3tnk escort x=64 → x=34; sentinel (108,20) → (52,20);
# forward markers x=56/x=64 → x=28/x=34. Safe-zone discs (5,20) /
# (5,10) / (5,30) preserved. Intended policy ENGAGE_X=24, plus an
# approach-via-home-y latitude (the tighter diagonal would otherwise
# expose the lead tank to e3 fire before the squad reaches the line).
P='combat-retreat-after-engagement'; C='reasoning'; M='arena 56x40 (6 obs)'; SZ='56x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(5,18-21)',
    '2× rocket infantry @(30,19/21) + 1× heavy tank @(34,20) + far sentinel @(52,20)',
    "Commander, this is a strike-and-retreat mission on a tight 56x40 arena. You are given 4 medium tanks in the western safe zone. The enemy firing line — 2 anti-tank rocket soldiers at (30, 19/21) plus 1 heavy tank set back at (34,20) — out-trades a head-on fight. Push east, focus-fire the two rockets, then pull back to the (5,20) safe disc. Preserving the force matters more than the kill count.",
    'At least 2 kills, then ≥3 tanks back in the (5,20) disc, ≥2 alive, within 4000 ticks.',
    'Fewer than 2 alive, or deadline (4000).',
    51, 4593)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank @(5,18-21)',
    '3× rocket infantry @(30, y=18/20/22) + 1× heavy tank @(34,20)',
    "Commander, this is a strike-and-retreat mission on a tight 56x40 arena. You are given 4 medium tanks in the western safe zone. The enemy firing line — 3 anti-tank rocket soldiers at (30, y=18/20/22) plus 1 heavy tank at (34,20) — out-trades a head-on fight. Strike, kill all three rockets, then disengage west to the (5,20) safe disc before the heavy collapses the column.",
    'At least 3 kills + ≥3 back in (5,20) disc, ≥3 alive, within 4000 ticks.',
    'Fewer than 3 alive, or deadline.',
    51, 4593)
add(P,'hard',C,M,SZ,'fit',T,
    '4× medium tank staged NORTH (5,8-11) or SOUTH (5,28-31) by seed',
    '2× rocket infantry @(30, 19/21) + 1× heavy tank @(34,20)',
    "Commander, this is a strike-and-retreat mission with seed-rotated staging. You are given 4 medium tanks in either the north (5, y=8..11) or south (5, y=28..31) safe zone. The enemy firing line at (30, 19/21) plus a heavy tank at (34,20) out-trades a head-on fight. Strike, kill the two rocket soldiers, then retreat to YOUR safe zone — not the wrong one.",
    'At least 2 kills + ≥3 in MATCHING safe-zone disc, ≥3 alive, within 4000 ticks.',
    'Fewer than 3 alive, or deadline.',
    51, 4593)

# 20. combat-rocket-soldier-anti-vehicle — [EDITED 2026-05-23] map
# shrunk rush-hour-arena 128x40 → tailored arena 64x40 per tier;
# base at x=8, enemy band at x=40 (Dragon range from x=35). POSTURE
# FIX: enemy heavies bumped from stance:0 HoldFire → stance:2 Defend
# (auto-fire on closest in-range attacker, never advance) so the
# counter-armor doctrine is actually load-bearing. Hard tier swaps
# the 3× Mammoth (4tnk) for 4× heavy tanks (3tnk) because Mammoth's
# secondary MammothTusk anti-infantry weapon (range 6.5) outranges
# Dragon (range 5), making the e3 counter physically unwinnable;
# 3tnk carries only the 105mm cannon (range 4.75) which Dragon
# outranges, so rocket-mass alpha-strike works.
P='combat-rocket-soldier-anti-vehicle'; C='reasoning'; M='arena 64x40'; SZ='64x40'
T='build, place_building, move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    'Starter base on the west (Construction Yard @(8,20), Power Plant, Infantry Barracks, War Factory, Service Depot) + $1800',
    '2× heavy tanks @(40,18/22) on Defend orders + far sentinel Construction Yard @(60,20)',
    "Commander, this is a counter-armour engagement. You are given a starter base on the west — Construction Yard, Power Plant, Infantry Barracks, War Factory, and Service Depot — plus $1800. Two enemy heavy tanks hold position at (40, 18..22) on Defend orders: they will auto-fire on the closest in-range attacker but won't chase. Rifle infantry can't scratch tank armour; the budget only buys two light tanks, which lose the attrition trade against the heavies. Train six rocket soldiers (anti-vehicle launchers) from the Infantry Barracks, advance east into Dragon range, kill at least one heavy, keep your Construction Yard, within about 50 turns.",
    'Train ≥6× rocket infantry AND ≥1 kill, ≥1 own unit alive, hold our Construction Yard, within 4500 ticks.',
    'Construction Yard lost, or deadline.',
    51, 4593)
add(P,'medium',C,M,SZ,'fit',T,
    'Starter base on the west (Construction Yard @(8,20), Power Plant, Infantry Barracks, War Factory, Service Depot) + $1800',
    '3× heavy tanks @(40, 18/20/22) on Defend orders (fog-concealed) + far sentinel fact',
    "Commander, this is a counter-armour engagement. You are given a starter base on the west — Construction Yard, Power Plant, Infantry Barracks, War Factory, and Service Depot — and $1800. Three enemy heavy tanks wait at (40, 18..22) under fog on Defend orders: they will return fire on whatever comes into range but won't chase. Rifle infantry can't dent armour; the budget only fields two light tanks, which lose the attrition trade. Train six rocket soldiers from the Infantry Barracks, advance east, kill at least two heavies, keep the yard, within about 50 turns.",
    '≥6× rocket infantry, ≥2 kills, ≥1 alive, hold Construction Yard, within 4500 ticks.',
    'Construction Yard lost, or deadline.',
    51, 4593)
add(P,'hard',C,M,SZ,'fit',T,
    'Starter base seed-rotated NORTH (8,12) or SOUTH (8,28) (Construction Yard + Power Plant + Barracks + War Factory + Service Depot + scout jeep) + $1800',
    '4× heavy tanks @(40, 16/19/21/24) on Defend orders + far sentinel fact',
    "Commander, this is a counter-armour engagement against a denser heavy formation. You are given a starter base — Construction Yard, Power Plant, Infantry Barracks, War Factory, Service Depot, and a scout jeep — staged either north (y=12) or south (y=28) by seed, and $1800. FOUR enemy heavy tanks hold the centre engagement column at (40, 16..24) on Defend orders: they will auto-fire on the closest in-range attacker but won't chase. Light tanks lose attrition against heavy armour; rifles can't dent it. Train six rocket soldiers from your barracks, march east from whichever band you started in, kill at least two heavies, keep your Construction Yard, within about 40 turns.",
    '≥6× rocket infantry, ≥2 kills, ≥1 alive, hold a Construction Yard, within 3600 ticks.',
    'Both Construction Yards lost, or deadline.',
    41, 3693)

# 21. combat-skirmish-then-disengage — [EDITED 2026-05-23] map shrunk
# rush-hour-arena 128x40 → tailored arena 64x40 per tier; jeeps at
# x=4-5, cluster at x=27-31. POSTURE FIX: enemy rifles bumped from
# stance:0 HoldFire → stance:1 ReturnFire so the strike does cost
# the raiders a trickle of damage and the "disengage to preserve"
# decision has real teeth. Hard tier keeps bot_type: hunt (overrides
# stance via st3 + Attack orders).
P='combat-skirmish-then-disengage'; C='action'; M='arena 64x40'; SZ='64x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× jeep @(4-5, 19-21) on HOLD-FIRE',
    '4× rifle infantry @(27/29, 19/21) on Return-Fire orders + far sentinel fact @(60,20)',
    "Commander, this is a skirmish-and-disengage raid. You are given 4 jeeps at the western staging point at (4,20). An enemy cluster of 4 riflemen holds a knot around (28,20). The riflemen are on Return-Fire orders — they only shoot back when shot at, never first. Your jeeps start on hold-fire so they will not engage on their own; you must order them in. Drive east, gun down at least three riflemen, then pull every survivor back into the 6-cell recovery disc around (4,20). Three kills, three jeeps back home, within about 50 turns.",
    '≥3 kills + ≥3 jeeps inside the (4,20) recovery disc, ≥3 alive, within 4500 ticks.',
    'All dead, or deadline.',
    52, 4683)
add(P,'medium',C,M,SZ,'fit',T,
    '4× jeep @(4-5, 19-21) on HOLD-FIRE',
    '6× rifle infantry @(27/29/31, 19/21) on Return-Fire orders + far sentinel fact',
    "Commander, this is a skirmish-and-disengage raid. You are given 4 jeeps at the western staging point at (4,20). The enemy cluster is bigger here — 6 riflemen knotted around (28,20), all on Return-Fire orders (they shoot back when shot at, never first). Your jeeps start on hold-fire. The cluster bleeds you every round you fight it, and mopping up all six eats the clock and leaves you stranded forward. Drive east, gun down three riflemen, then pull every survivor back into the 6-cell recovery disc around (4,20), keeping three jeeps alive, within about 50 turns.",
    '≥3 kills + ≥3 in (4,20) recovery disc, ≥3 alive, within 4500 ticks.',
    'All dead, or deadline.',
    52, 4683)
add(P,'hard',C,M,SZ,'fit',T,
    '4× jeep seed-staged NORTH (4,10) or SOUTH (4,30), all HOLD-FIRE',
    '6× rifle infantry @(27/29/31, 19/21) — HUNT BOT (st3 + Attack orders advance them west)',
    "Commander, this is a skirmish-and-disengage raid on a hostile cluster. You are given 4 jeeps at either the north staging point (4,10) or the south (4,30) by seed. An enemy cluster of 6 riflemen sits at (28,20) and is actively HUNTING — surviving riflemen advance on whichever of your jeeps is closest. Your jeeps start on hold-fire. Drive east, gun down three riflemen, then pull every survivor back into the 6-cell recovery disc around YOUR matching staging point — either (4,10) or (4,30). Three kills, three jeeps in your recovery disc, within about 50 turns. A slow retreat lets the hunters catch you.",
    '≥3 kills + ≥3 in EITHER spawn-corner recovery disc, ≥3 alive, within 4500 ticks.',
    'All dead, or deadline.',
    52, 4683)

# 22. combat-stance-mgmt-attack — [EDITED 2026-05-23] briefing rewrite
# to match the realised YAML (rush-hour-arena 128x40; defenders on
# stance:1 ReturnFire; scatter on stance:0 HoldFire across the eastern
# half). Capability under test: hunt-authorisation — escalate the
# formation from ReturnFire to AttackAnything so the engine's hunt
# path advances each tank to the scatter.
P='combat-stance-mgmt-attack'; C='action'; M='rush-hour-arena'; SZ='128x40'
T='observe, move_units, attack_unit, attack_move, stop, set_stance'
add(P,'easy',C,M,SZ,'fit',T,
    'Construction Yard @(5,20) + 4× medium tank @(10-11, 19-21) on Return-Fire orders',
    '4× rifle infantry scattered @(40,15), (60,25), (80,18), (95,22) on hold-fire',
    "Commander, this is a hunt-authorisation drill. You are given a Construction Yard at (5,20) and 4 medium tanks at (10-11, 19-21) on Return-Fire orders — they only open fire after being shot at, and none of the enemy is in range to provoke them. Four enemy riflemen are scattered across the eastern half of the map at (40,15), (60,25), (80,18), and (95,22), none aggressive — they will not come to you. Change the tanks' orders to Attack-Anything and the engine will pathfind each kill on its own. Kill three of the four, keep the Construction Yard, hold at least 3 of 4 tanks alive, within about 50 turns.",
    'Hold our Construction Yard, ≥3 kills, ≥3 tanks alive, within 4500 ticks.',
    'Construction Yard lost, all dead, or deadline.',
    60, 5403)
add(P,'medium',C,M,SZ,'fit',T,
    'Construction Yard @(5,20) + 4× medium tank @(10-11, 19-21) on Return-Fire orders',
    '3× rifle infantry + 1× light tank scattered across the east on hold-fire',
    "Commander, this is a hunt-authorisation drill. You are given a Construction Yard at (5,20) and 4 medium tanks at (10-11, 19-21) on Return-Fire orders — they only open fire after being shot at. The enemy is a scattered force across the eastern half: 3 riflemen at (40,15), (60,28), (80,12), and 1 light tank at (95,22). They are all on hold-fire and will not come to you. Change the tanks' orders to Attack-Anything; the engine will pathfind each kill on its own. Four kills, Construction Yard standing, at least 3 of 4 tanks alive, within about 50 turns.",
    'Hold Construction Yard, ≥4 kills, ≥3 alive, within 4500 ticks.',
    'Construction Yard lost, or deadline.',
    60, 5403)
add(P,'hard',C,M,SZ,'fit',T,
    'Construction Yard @(5,20) + 4× medium tank in NORTH (y=14-16) or SOUTH (y=24-26) band by seed on Return-Fire',
    '3× rifle infantry + 1× light tank scattered on hold-fire',
    "Commander, this is a hunt-authorisation drill with seed-varied staging. You are given a Construction Yard at (5,20) and 4 medium tanks on Return-Fire orders, staged on either the NORTH band (y=14-16) or the SOUTH band (y=24-26) depending on seed — read your own position from the map. The enemy is a scattered force across the eastern half: 3 riflemen and 1 light tank, all on hold-fire and not aggressive. A memorised hunt path will not generalise: change the tanks' orders to Attack-Anything and let the engine pathfind the kills on its own. Four kills, Construction Yard intact, at least 3 of 4 tanks alive, within about 50 turns.",
    'Hold a Construction Yard, ≥4 kills, ≥3 alive, within 4500 ticks.',
    'Construction Yard lost, or deadline.',
    60, 5403)

# 23. combat-suicide-charge-mission — [EDITED 2026-05-23] map shrunk
# 96x40 → 56x40 per tier; objective fact (86,20)→(50,20); picket
# x=82..85 → x=46..49; sentinel (92,4)→(53,4). Hard tier chokepoint
# walls retuned to mid-map x=24..27 and picket dropped to 2× e3
# (medium kept 3× e3) so the chokepoint funnel — the load-bearing
# hard escalation — doesn't over-concentrate the picket's fire on
# the emerging strike package. No posture defect: defenders use
# bot_type: guard with stance:2 (lunge within ~16, snap back past
# ~18) which matches the briefing's "guards lunge" premise.
P='combat-suicide-charge-mission'; C='reasoning'; M='arena 56x40'; SZ='56x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '6× medium tank + 4× rocket infantry strike package @(4-6, 16-24)',
    '2× rocket infantry guards @(46, 17/23) + 1× Mammoth tank @(49,20) (guard bot, lunge ~16) + enemy Construction Yard @(50,20) + sentinel fact @(53,4)',
    "Commander, this is a forlorn-hope strike on a defended enemy Construction Yard. You are given a strike package on the west edge: six medium tanks and four rocket soldiers (anti-vehicle Dragon launchers). The objective is the enemy Construction Yard at (50, 20) on the far east — it must fall. The yard is guarded by a light picket (2 rocket soldiers at x=46 and a Mammoth tank at x=49) on guard orders: they hold their post but lunge at any enemy within roughly 16 cells then snap back. Keeping the strike package alive is NOT possible and NOT required. Commit every unit straight at the yard, focus-fire through the defenders, accept the losses, and raze the yard within about 60 turns.",
    'Destroy the enemy Construction Yard at (50,20), within 5400 ticks. (Survival not required.)',
    'Deadline only (5401).',
    62, 5583)
add(P,'medium',C,M,SZ,'fit',T,
    '6× medium tank + 4× rocket infantry strike package @(4-6, 16-24)',
    '3× rocket infantry guards @(46, 16/20/24) + 1× Mammoth tank @(49,20) (guard bot) + enemy Construction Yard @(50,20) + sentinel fact',
    "Commander, this is a forlorn-hope strike on a defended enemy Construction Yard. You are given a strike package on the west edge: six medium tanks and four rocket soldiers. The objective is the enemy Construction Yard at (50, 20) on the far east. This picket is heavier — three rocket soldiers spread across y=16/20/24 at x=46 with a Mammoth tank behind at x=49, all on guard orders. Their Dragons outrange you in any standoff. Do not try to preserve the force: commit every unit, focus-fire through the wall, and raze the yard within about 72 turns.",
    'Destroy the enemy Construction Yard, within 6400 ticks.',
    'Deadline (6401).',
    73, 6573)
add(P,'hard',C,M,SZ,'fit',T,
    'Strike package split NORTH (4-6, 8-12) or SOUTH (4-6, 28-32) by seed (6× 2tnk + 4× e3 each)',
    '2× rocket infantry @(46, 17/23) + 1× Mammoth tank pulled back to @(52,20) (guard bot) + enemy Construction Yard @(50,20) + sentinel fact; water cordon at mid-map (x=24..27) funnels both axes through the y=15..25 corridor',
    "Commander, this is a two-prong forlorn-hope strike. You are given the same ten-unit strike package — six medium tanks and four rocket soldiers — staging in either the north band (around y=10) or the south band (around y=30), chosen by seed. A water cordon at mid-map leaves only a narrow y=15..25 corridor open, so both spawn bands must funnel through that one pinch before reaching the picket. The enemy yard is at (50, 20), guarded by two rocket soldiers at x=46 and a Mammoth tank pulled back to x=52, all on guard orders. Halting short of the corridor or trying to keep the force alive will both miss the clock. Commit every unit all-in along whichever axis your seed opens, eat the casualties, and raze the yard within about 80 turns.",
    'Destroy the enemy Construction Yard, within 7200 ticks.',
    'Deadline (7201).',
    82, 7383)

# 24. combat-tank-vs-tank-engagement — [EDITED 2026-05-23] briefing
# rewrite (self-contained per level, explicit coords); enemy line is
# 3-vs-3 (easy/hard) and 4-vs-3 on medium per the recalibration; hard
# tier uses seed-driven NORTH/SOUTH staging at x=30 in [y=11-13] vs
# [y=27-29].
P='combat-tank-vs-tank-engagement'; C='action'; M='rush-hour-arena'; SZ='128x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    'Construction Yard @(4,20) + 3× medium tank @(30,19-21)',
    '3× medium tanks at (50,15), (51,20), (50,25) + far Construction Yard',
    "Commander, this is a mirror tank fight. You are given a Construction Yard at (4,20) and 3 medium tanks bunched at (30, 19-21). The enemy fields 3 medium tanks along a vertical line at (50, 15), (51, 20), and (50, 25). Close to cannon range and concentrate fire on one enemy tank at a time, starting with the nearest — three tanks shooting one target will drop it fast, and every enemy lost cuts their firepower. Do not drive straight into their midst; that bunches you in their crossfire and wipes the squad. Three kills, one tank alive, Construction Yard intact, within about 27 turns.",
    '≥3 kills, ≥1 alive, hold Construction Yard, within 2400 ticks.',
    'All dead, Construction Yard lost, or deadline (2401).',
    30, 2700)
add(P,'medium',C,M,SZ,'fit',T,
    'Construction Yard @(4,20) + 3× medium tank @(30,19-21)',
    '4× medium tanks at (50,14), (51,18), (50,22), (51,26)',
    "Commander, this is a mirror tank fight against superior numbers. You are given a Construction Yard at (4,20) and 3 medium tanks bunched at (30, 19-21). The enemy fields 4 medium tanks spread along the eastern line at (50, 14), (51, 18), (50, 22), and (51, 26) — you are out-gunned 4-vs-3. Close to cannon range, concentrate fire on one target at a time, and work down the line so each kill cuts their firepower. Driving straight into the cluster bunches you in their crossfire and wipes the squad. Three kills, two tanks alive, Construction Yard intact, within about 27 turns.",
    '≥3 kills, ≥2 alive, hold Construction Yard, within 2400 ticks.',
    'Construction Yard lost or fewer than 2 alive, or deadline.',
    30, 2700)
add(P,'hard',C,M,SZ,'fit',T,
    '2× Construction Yards + 3× medium tank @x=30 in NORTH (y=11-13) or SOUTH (y=27-29) corridor by seed',
    '3× medium tanks at (50,15), (51,20), (50,25)',
    "Commander, this is a mirror tank fight on a tight clock with seed-varied staging. You are given a Construction Yard at (4,20) and 3 medium tanks bunched at x=30, in either the NORTH corridor (y=11-13) or the SOUTH corridor (y=27-29) depending on seed — read your own position from the map. The enemy holds a fixed vertical line of 3 medium tanks at (50, 15), (51, 20), and (50, 25). You have about 14 turns: close to cannon range and concentrate fire on one enemy tank at a time. A brute drive into the line bunches you in crossfire and wipes the squad; stalling busts the clock. Three kills, one tank alive, Construction Yard intact.",
    '≥3 kills, ≥1 alive, hold a Construction Yard, within 1200 ticks.',
    'All dead, or deadline (1201).',
    15, 1353)

# 25. combat-tanya-vs-rush — [EDITED 2026-05-23] map shrunk from
# rush-hour-arena 128x40 → tailored arena 32x40 (Tanya at (12,20),
# enemy rush at x=16-19; far enemy fact at (28,20)). The rush
# engagement is a single localised cell — 128 cells of arena was
# wasted padding.
P='combat-tanya-vs-rush'; C='action'; M='arena 32x40'; SZ='32x40'
T='move_units, attack_unit, attack_move, stop, set_stance'
add(P,'easy',C,M,SZ,'fit',T,
    '1× Tanya @(12,20) on hold-fire (elite commando, fast, tough, pistol one-shots riflemen)',
    '4× rifle infantry in a line at x=16-19, y=20 (HUNTER: st3 rush)',
    "Commander, this is a hero engagement. You are given Tanya at (12,20) — an elite commando: fast, tough, with a pistol that one-shots riflemen. She starts on hold-fire (she will NOT engage even if shot at). The enemy rushes from the east on attack-anything stance: 4 riflemen in a line at x=16-19, y=20. Lift Tanya's hold-fire order and attack the riflemen — at point-blank her pistol cuts down all four faster than they can dent her 150k HP. Sitting idle lets them surround her and she dies. Four kills, Tanya alive, within about 30 turns.",
    '≥4 kills, Tanya alive, within 2700 ticks.',
    'Tanya dead, or deadline.',
    30, 2700)
add(P,'medium',C,M,SZ,'fit',T,
    '1× Tanya @(12,20) on hold-fire',
    '6× rifle infantry in a 2×3 block at (16-17, 19-21) (HUNTER: st3 rush)',
    "Commander, this is a hero engagement against a denser rush. You are given Tanya at (12,20) — an elite commando: fast, tough, with a pistol that one-shots riflemen. She starts on hold-fire (she will NOT engage even if shot at). The enemy rushes from the east in a 2×3 block on attack-anything stance: 6 riflemen in two tight ranks at (16-17, 19-21). Lift Tanya's hold-fire order and attack the riflemen at point-blank — her pistol cuts down the whole pack faster than they can dent her HP. Sitting idle lets them gang up on her. Six kills, Tanya alive, within about 30 turns.",
    '≥6 kills, Tanya alive, within 2700 ticks.',
    'Tanya dead, or deadline.',
    30, 2700)
add(P,'hard',C,M,SZ,'fit',T,
    '1× Tanya @(12,20) on hold-fire',
    '6× rifle infantry in 2×3 block at NORTH (y=15-17) OR SOUTH (y=23-25) by seed (HUNTER: st3 rush)',
    "Commander, this is a hero engagement with seed-varied rush direction. You are given Tanya at (12,20) — an elite commando: fast, tough, with a pistol that one-shots riflemen. She starts on hold-fire (she will NOT engage even if shot at). The enemy rushes from one of two corridors depending on seed: 6 riflemen in a 2×3 block at the NORTH band (y=15-17) OR at the SOUTH band (y=23-25), all at x=16-17. Read the rush direction from the map, lift Tanya's hold-fire order, and attack the riflemen at point-blank. Sitting idle lets them gang up. Six kills, Tanya alive, within about 30 turns.",
    '≥6 kills, Tanya alive, within 2700 ticks.',
    'Tanya dead, or deadline.',
    30, 2700)

# 26. combat-target-priority-highvalue — [EDITED 2026-05-23] briefing
# rewrite. The realised YAML uses e1 chaff + e3 rocket soldiers (NOT a
# proc refinery — the audit's high-value asset is the rocket-soldier
# trio, the threat-priority target). Briefings reflect that.
P='combat-target-priority-highvalue'; C='action'; M='arena 112x40'; SZ='112x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    '4× medium tank @(48,18-21) + Construction Yard @(4,20)',
    '9× rifle infantry screen @x=66 + 3× anti-tank rocket soldiers (e3) @x=69',
    "Commander, this is a target-priority strike. You are given a Construction Yard at (4,20) and 4 medium tanks at (48, 18-21). The enemy is a mixed cluster to the east: a screen of 9 riflemen at x=66 backed by 3 anti-tank rocket soldiers at x=69. The riflemen barely scratch Heavy armour; the rocket soldiers carry Dragon launchers and are the real threat. Concentrate all four tanks' fire on one rocket soldier at a time — four-on-one drops it in a turn or two — then the next, then the third, then mop up the rifles. Engaging the chaff first leaves all three rockets firing through the whole mop-up. Twelve kills, three tanks alive, Construction Yard intact, within about 30 turns.",
    '≥12 kills, ≥3 tanks alive, hold Construction Yard, within 2700 ticks.',
    'Fewer than 3 alive, Construction Yard lost, or deadline.',
    30, 2700)
add(P,'medium',C,M,SZ,'fit',T,
    '4× medium tank @(48,18-21) + Construction Yard @(4,20)',
    '12× rifle infantry screen @x=66 + 3× anti-tank rocket soldiers @x=69',
    "Commander, this is a target-priority strike against a heavier cluster. You are given a Construction Yard at (4,20) and 4 medium tanks at (48, 18-21). The enemy is 12 riflemen in a tight screen at x=66 backed by 3 anti-tank rocket soldiers at x=69. The riflemen barely scratch Heavy armour; the rocket soldiers (Dragon launchers) are the real threat. Concentrate all four tanks' fire on one rocket soldier at a time, then the next, then the third, then sweep the rifles. Starting on the chaff leaves all three rockets firing through the entire mop-up — you bleed tanks and run out of time. Fifteen kills, three tanks alive, Construction Yard intact, within about 30 turns.",
    '≥15 kills, ≥3 alive, hold Construction Yard, within 2700 ticks.',
    'Fewer than 3 alive, Construction Yard lost, or deadline.',
    30, 2700)
add(P,'hard',C,M,SZ,'fit',T,
    '4× medium tank @x=48 in NORTH (y=10-13) or SOUTH (y=27-30) corridor by seed + Construction Yard',
    '12× rifle infantry screen @x=66 + 3× anti-tank rocket soldiers @x=69',
    "Commander, this is a target-priority strike with seed-varied staging. You are given a Construction Yard and 4 medium tanks at x=48, in either the NORTH corridor (y=10-13) or the SOUTH corridor (y=27-30) depending on seed — read your own position from the map. The enemy is 12 riflemen in a tight screen at x=66 backed by 3 anti-tank rocket soldiers at x=69. The riflemen barely scratch Heavy armour; the rocket soldiers (Dragon launchers) are the real threat. Concentrate all four tanks' fire on one rocket soldier at a time — four-on-one drops it fast — then the next, then the third, then sweep the rifles. Starting on the chaff leaves all three rockets firing through the whole engagement. Fifteen kills, three tanks alive, Construction Yard intact, within about 30 turns.",
    '≥15 kills, ≥3 alive, hold Construction Yard, within 2700 ticks.',
    'Fewer than 3 alive, Construction Yard lost, or deadline.',
    30, 2700)

# 27. combat-vehicle-vs-infantry-counter — [EDITED 2026-05-23] map
# shrunk from rush-hour-arena 128x40 → tailored arena 64x40. The base
# spans x=8-16, the enemy cluster sits at x=40, the far enemy fact at
# (60,20). Briefings rewritten as self-contained per level (mission
# framing → forces given → objective).
P='combat-vehicle-vs-infantry-counter'; C='reasoning'; M='arena 64x40'; SZ='64x40'
T='build, place_building, move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    'Full starter base @(10-18, 16-22): Construction Yard, Power Plant, Infantry Barracks, War Factory, Service Depot + 1× scout jeep (HoldFire) + $2550 cash',
    '8× rifle infantry cluster around (40,20) + far Construction Yard @(60,20)',
    "Commander, this is a hard-counter selection mission. You are given a full starter base on the west — Construction Yard, Power Plant, Infantry Barracks, War Factory, Service Depot — plus a scout jeep on hold-fire orders (it scouts, it does not fight). You have $2550 cash. The enemy is a cluster of 8 pure rifle infantry around (40,20); tanks crush riflemen easily (Heavy armour soaks small-arms fire). Your budget buys exactly one dominant composition: 3 medium tanks at $850 each ($2550 total). Mass rockets waste cost-per-effect against soft targets; matching with own rifles is a 1:1 trade with no advantage. Build the three tanks from the War Factory and crush the cluster. Hold the Construction Yard, ≥6 kills, within about 60 turns.",
    'Train ≥3× medium tank + ≥6 kills + hold our Construction Yard, within 5400 ticks.',
    'Construction Yard lost, all dead, or deadline.',
    60, 5403)
add(P,'medium',C,M,SZ,'fit',T,
    'Full starter base on the west + scout jeep + $2550 cash',
    '12× rifle infantry cluster around (40,20) + far Construction Yard',
    "Commander, this is a hard-counter selection mission. You are given a full starter base on the west — Construction Yard, Power Plant, Infantry Barracks, War Factory, Service Depot — plus a scout jeep on hold-fire orders. You have $2550 cash. The enemy holds the centre with 12 rifle infantry around (40,20) under fog of war; scout with the jeep to verify they carry no anti-tank threats hiding among them. Three medium tanks ($2550 exactly) walk through small-arms fire; rockets waste cost-per-effect and own rifles are a 1:1 trade with no advantage. Build the trio, kill eight, hold the Construction Yard, within about 60 turns.",
    'Train ≥3× medium tank + ≥8 kills + hold Construction Yard, within 5400 ticks.',
    'Construction Yard lost, or deadline.',
    60, 5403)
add(P,'hard',C,M,SZ,'fit',T,
    'Full starter base spawned NORTH (y=12) or SOUTH (y=28) by seed + scout jeep + $2550 cash',
    '12× rifle infantry cluster around (40,20) + far Construction Yard',
    "Commander, this is a hard-counter selection mission with seed-varied staging. You are given a full starter base — Construction Yard, Power Plant, Infantry Barracks, War Factory, Service Depot — plus a scout jeep on hold-fire orders. The base spawns either NORTH (y=12) or SOUTH (y=28) depending on seed — read your own position from the map. You have $2550 cash. The enemy holds the central lane around (40,20) with 12 rifle infantry under fog of war; scout with the jeep to verify they carry no anti-tank threats. Three medium tanks ($2550 exactly) walk through small-arms fire; rockets and own rifles both lose the trade. Build the trio, kill eight, hold a Construction Yard, within about 60 turns.",
    '≥3× medium tank total + ≥8 kills + hold a Construction Yard, within 5400 ticks.',
    'Both Construction Yards lost, or deadline.',
    60, 5403)

# 28. harass-response-preserve — [EDITED 2026-05-23] briefing rewrite
# (self-contained per level with mission framing → forces given →
# objective; hold-fire mechanic explained inline; "do not chase east"
# trap surfaced).
P='harass-response-preserve'; C='action'; M='rush-hour-arena'; SZ='128x40'
T='move_units, attack_unit, attack_move, stop'
add(P,'easy',C,M,SZ,'fit',T,
    'Ore Refinery @(16,20) + 5× rifle-infantry garrison @(13-14, 18-22) on HOLD-FIRE',
    '1× rifle-infantry probe @(22,20) on patrol bot (oscillates 14-30 along y=20)',
    "Commander, this is a base-defence response. You are given an Ore Refinery at (16,20) and a 5-man rifle-infantry garrison clustered around it at (13-14, 18-22), all on hold-fire — they will not engage unless you order them to. A single enemy rifleman is patrolling inside the perimeter, oscillating between (14,20) and (30,20); every west pass it chips the Refinery. Keep the garrison ON the Refinery and order them to attack the probe as it swings back into range — do not chase east, that leaves the Refinery undefended. One kill, Refinery still standing, at least four garrison alive, no more than two losses, within about 50 turns.",
    '≥1 kill, ≥4 garrison alive, ≤2 losses, hold Refinery, within 4500 ticks.',
    'More than 2 lost, Refinery lost, or deadline.',
    55, 5043)
add(P,'medium',C,M,SZ,'fit',T,
    'Ore Refinery @(16,20) + 5× rifle-infantry garrison on HOLD-FIRE',
    '2× rifle-infantry probes on patrol bot (oscillating arc 14-30, y=20)',
    "Commander, this is a base-defence response. You are given an Ore Refinery at (16,20) and a 5-man rifle-infantry garrison clustered around it at (13-14, 18-22), all on hold-fire. Two enemy riflemen are patrolling the same arc inside the perimeter, oscillating between (14,20) and (30,20); they chip the Refinery on every west pass. Hold the garrison ON the Refinery and order both probes engaged as they swing into range — chasing east leaves the Refinery undefended. Two kills, Refinery alive, five garrison alive, at most one loss, within about 50 turns.",
    '≥2 kills, ≥5 alive, ≤1 loss, hold Refinery, within 4500 ticks.',
    'More than 1 lost, Refinery lost, or deadline.',
    55, 5043)
add(P,'hard',C,M,SZ,'fit',T,
    'Ore Refinery @(16,20) + Barracks @(18,20) + 5× rifle-infantry garrison NORTH (y=17-19) or SOUTH (y=21-23) by seed, on HOLD-FIRE',
    '3× rifle-infantry probes on patrol bot (anchor 26,20, oscillating 18-34)',
    "Commander, this is a base-defence response with twin infrastructure and seed-varied staging. You are given an Ore Refinery at (16,20) and an adjacent Infantry Barracks at (18,20). A 5-man rifle-infantry garrison stages on either the north band (y=17-19) or the south band (y=21-23) depending on seed — read your own position from the map. Either band can cover both buildings. The garrison is on hold-fire — they will not engage unless ordered. Three enemy riflemen are probing the central latitude (anchor x=26, oscillating 18..34); they chip both buildings as they swing west. Hold the asset footprint and engage each probe in range — chasing east or pulling back loses a building. Three kills, both buildings alive, all five garrison alive, ZERO losses, within about 50 turns.",
    '≥3 kills, ≥5 alive, ZERO losses, hold Refinery AND Barracks, within 4500 ticks.',
    'Any loss, infra lost, or deadline.',
    55, 5043)

# 29. rush-hour — [EDITED 2026-05-23] briefing rewrite + max_turns 41
# set (was unset; the 3603-tick within_ticks deadline now bites
# because 3603 ≤ 93 + 90·40 = 3693, so timeout is a real LOSS not a
# DRAW).
P='rush-hour'; C='action'; M='rush-hour-arena'; SZ='128x40'
T='move_units, attack_unit, attack_move, stop_units'
add(P,'easy',C,M,SZ,'fit',T,
    '24× medium tank + 8× jeep deployed at all four corners (sp0..sp3, 6 units per corner)',
    '22× rifle infantry scattered across 11 sites on north, mid, south rows, hidden by fog',
    "Commander, this is a reconnaissance-and-destroy sweep. You are given four armies, one at each corner of the 128×40 arena — 24 medium tanks and 8 jeeps in total (six units per corner). The enemy is 22 rifle infantry scattered across the arena in small groups, all hidden by fog of war on the north, mid, and south rows. Spread the armies, reveal the fog as you move, find and destroy the enemy. Score at least seven kills, reveal at least 18% of the map, lose no more than five of your own, within about 41 turns. Sitting still or marching down a single row fails both bars.",
    '≥7 kills + ≥18% of the map revealed + ≤5 losses, within 3603 ticks.',
    'All dead, more than 5 lost, or deadline.',
    41, 3603)
add(P,'medium',C,M,SZ,'fit',T,
    '24× medium tank + 8× jeep deployed at all four corners',
    '22× rifle infantry scattered across 11 sites on north, mid, south rows, hidden by fog',
    "Commander, this is a wider reconnaissance-and-destroy sweep. You are given four corner armies — 24 medium tanks and 8 jeeps in total. The enemy is 22 rifle infantry scattered across the 128×40 arena in small groups, all hidden by fog of war on the north, mid, and south rows. Spread the armies, reveal the fog, and find and destroy the enemy. Score at least nine kills, reveal at least 28% of the map, lose no more than four of your own, within about 41 turns. A single-row pass tops out at eight kills — clearing both the north and south bands is required.",
    '≥9 kills + ≥28% revealed + ≤4 losses, within 3603 ticks.',
    'All dead, more than 4 lost, or deadline.',
    41, 3603)
add(P,'hard',C,M,SZ,'fit',T,
    '24× medium tank + 8× jeep deployed at all four corners (compass-only briefing)',
    '22× rifle infantry scattered across 11 sites on north, mid, south rows, hidden by fog',
    "Commander, this is a reconnaissance-and-destroy sweep across the whole 128×40 arena. Your team round-robins between four spawn corners by seed (compass-only briefing): you may stage at any of NW, NE, SW, or SE — read your own starting corner from the map. You are given 24 medium tanks and 8 jeeps in total (six units per corner). The enemy is 22 rifle infantry scattered across the arena in small groups, all hidden by fog of war on the north, mid, and south rows; localise them on the minimap yourself. Score at least ten kills, reveal at least 32% of the map, lose no more than four of your own, within about 41 turns. Only a real spread across all three rows clears the bar.",
    '≥10 kills + ≥32% revealed + ≤4 losses, within 3603 ticks.',
    'All dead, more than 4 lost, or deadline.',
    41, 3603)

# -- enemy posture overlay --
# Posture is scanned from the YAML enemy actors (stance + bot_type) and
# checked against what the briefing implies the enemy should do.
# Values: 'static' (st2 Defend) | 'passive' (st0 HoldFire) | 'reactive' (st1 ReturnFire)
#        | 'hunter' (st3 AttackAnything OR bot_type=hunt) | 'guard' (bot_type=guard, lunge-then-leash)
#        | 'patrol' (bot_type=patrol) | 'mixed'.
# posture_issue: '' if posture matches the briefing premise; otherwise a short note.
POSTURE = {
    'action-multiunit-coordination':       ('static (st2 pickets + landmark buildings)', ''),
    'action-sequenced-execution':          ('static (st2 pickets + landmark buildings)', ''),
    'combat-attack-from-behind-fog':       ('static (st2 dug-in line)', ''),
    'combat-bait-counter-attack':          ('guard (bot lunges at nearby enemy then leashes back)', ''),
    'combat-divide-and-conquer':           ('hunter (st3 AttackAnything; advances when baited)', ''),
    'combat-flanking-attack':              ('static (st2 dug-in line)', ''),
    'combat-focus-fire-priority':          ('static (st2 mixed cluster)', ''),
    'combat-formation-tank-wedge':         ('static (st2 line, single 1tnk st1 blocker)', ''),
    'combat-harass-aggro-commit':          ('static defender (st2 heavy tank + harvesters)', ''),
    'combat-harass-balanced-hit-and-run':  ('guard heavy + st0 workers (heavy lunges within ~16)', ''),
    'combat-heli-flank':                   ('passive (st0 infantry behind impassable wall)', ''),
    'combat-hold-chokepoint':              ('HUNTER (hunt bot + st3) — actively attacks chokepoint', ''),
    'combat-kite-and-pull':                ('hunter (hunt bot + st2 heavy)', ''),
    'combat-kite-jeep-vs-tank':            ('hunter (hunt bot + st2 heavy)', ''),
    'combat-naval-shore-strike':           ('passive (st0 shore infantry)', ''),
    'combat-pincer-coordination':          ('static (st2 central cluster)', ''),
    'combat-prevent-retreat':              ('static (st2 column)', 'DEFECT: briefing says "infantry will run east the moment we strike", but st2 = Defend (never moves). Enemy will NOT flee. Either rewrite briefing (no escape required) or change stance to st3 + add east-of-cluster waypoint so they advance east.'),
    'combat-protect-vip-escort':           ('hunter (guard bot + st3 belts)', ''),
    'combat-retreat-after-engagement':     ('guard (bot lunges in range; st2 + guard makes head-on unwinnable)', ''),
    'combat-rocket-soldier-anti-vehicle':  ('static (st2 heavy tanks — Defend, auto-fire in range)', ''),
    'combat-skirmish-then-disengage':      ('reactive (st1 rifle cluster — ReturnFire, fire back after being shot at)', ''),
    'combat-stance-mgmt-attack':           ('passive (st0 infantry) — INTENTIONAL: isolates the agent-stance-flip test', ''),
    'combat-suicide-charge-mission':       ('guard (bot + st2; defenders lunge as strike closes)', ''),
    'combat-tank-vs-tank-engagement':      ('static (st2 enemy tanks)', 'NOTE: mirror fight; st3 would be more realistic (active mirror engagement) but st2 is acceptable for the focus-fire test.'),
    'combat-tanya-vs-rush':                ('HUNTER (st3 rifle rush; advance + fire)', ''),
    'combat-target-priority-highvalue':    ('static (st2 dense screen)', ''),
    'combat-vehicle-vs-infantry-counter':  ('hunter (st3 infantry mass; advances on base)', ''),
    'harass-response-preserve':            ('patrol probe (patrol bot + st2)', ''),
    'rush-hour':                           ('reactive (st1 ReturnFire; scattered, hidden by fog)', ''),
}
for r in R:
    p, pi = POSTURE.get(r['pack'], ('', 'UNCATEGORISED'))
    r['enemy_posture'] = p
    r['posture_issue'] = pi

# -- emit CSV --
fields = ['pack','level','capability','map_name','map_size','map_fit','tools',
          'agent_force','enemy_force','enemy_posture','posture_issue',
          'briefing_RA','win_condition','lose_condition',
          'max_turns','tick_budget']
with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    w.writeheader()
    for r in R:
        w.writerow(r)
print(f'Wrote {len(R)} rows to {OUT}')
