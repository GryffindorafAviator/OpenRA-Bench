"""Builds audits/family7_procedure.csv — Family-7 (Procedure / Strict /
Maintenance / Robustness) manual audit.

One row per (pack, level). 22 packs × 3 levels = 66 rows. Inherits the
F1 prose/map conventions (§1-10) and the F1 no-solution-leak rule
(§9.5). Adds F7-specific columns `forbidden_tools` and
`scheduled_events` to capture the procedural axis the family probes.

Family-7 packs test procedural compliance and adaptive robustness:
- proc-* / strict-* — follow an ordered checklist, honour a tool
  allowlist or `forbidden_tools` ban, ignore distractor tools, dispatch
  on an IF/ELSE condition.
- maint-* — repair / sell the right buildings under a maintenance
  budget.
- rob-* — re-plan when an exogenous mid-episode event changes the
  operating conditions (scheduled spawn / destroy / shorten_deadline,
  pre-placed surprise wave, structural establishment shortfall).

Briefings are SELF-CONTAINED, officer-style, no-solution-leak (F1
§9.5): they describe the procedure / situation, NOT the policy. Where
a pack pre-discloses the mid-episode rule in its own YAML briefing
(rob-deadline-shortened-midway tells the model "the deadline gets cut
at tick 1000"), the audit briefing preserves the disclosure as part
of the operating conditions, since the test is the REACTION not the
DISCOVERY.
"""
import csv
from pathlib import Path

OUT = Path(__file__).parent / 'family7_procedure.csv'
R = []


def add(pack, level, cap, map_name, map_size, map_fit, tools,
        forbidden_tools, scheduled_events, agent, enemy,
        briefing, win, lose, max_turns, tick_budget,
        posture='', posture_issue=''):
    R.append(dict(
        pack=pack, level=level, capability=cap, map_name=map_name,
        map_size=map_size, map_fit=map_fit, tools=tools,
        forbidden_tools=forbidden_tools,
        scheduled_events=scheduled_events,
        agent_force=agent, enemy_force=enemy,
        enemy_posture=posture, posture_issue=posture_issue,
        briefing_RA=briefing, win_condition=win, lose_condition=lose,
        max_turns=max_turns, tick_budget=tick_budget,
    ))


# ──────────────────────────────────────────────────────────────────
# proc-* packs (10)
# ──────────────────────────────────────────────────────────────────

# 1. proc-checklist-no-deviation
P = 'proc-checklist-no-deviation'
C = 'action'
M = 'rush-hour-arena'
SZ = '128x40'
T = 'move_units, observe, stop'
add(P, 'easy', C, M, SZ, 'fit', T, '[]', 'none',
    '3× jeep @(6,18-22) on HoldFire',
    'No enemy; map is empty. Distant landmarks at the listed checkpoints.',
    "Commander, this is an ordered-checklist mission with only the movement tools (move_units, observe, stop). Visit three named checkpoints STRICTLY IN ORDER from your west-edge jeep column: S1 at the north edge (40,4), then S2 at the south edge (75,36), then S3 east-centre (118,20). Reaching a later checkpoint before its predecessor does not count and does not advance the chain. Idling runs out the clock. Within about 33 turns.",
    'Latched waypoint chain S1 → S2 → S3 (radius 5), within 2900 ticks.',
    'No jeeps left, or deadline (2901 ticks).',
    34, 3063)
add(P, 'medium', C, M, SZ, 'fit', T, '[]', 'none',
    '3× jeep @(6,18-22) on HoldFire',
    'No enemy; four landmark checkpoints at the listed cells.',
    "Commander, same ordered-checklist mission. Visit FOUR named checkpoints in order from your west-edge jeep column: S1 (40,4) north, S2 (75,36) south, S3 (95,4) NE, S4 (118,36) SE corner. A later checkpoint reached early does not advance the chain; a skipped step is unrecoverable. Within about 43 turns.",
    'Latched chain S1 → S2 → S3 → S4 (radius 5), within 3800 ticks.',
    'No jeeps left, or deadline (3801 ticks).',
    44, 3963)
add(P, 'hard', C, M, SZ, 'fit', T, '[]', 'none',
    '3× jeep seed-spawned at FAR-WEST x=6 (sp0) or NEAR-WEST x=20 (sp1), HoldFire',
    'No enemy; five landmark checkpoints.',
    "Commander, same ordered-checklist mission with a seed-chosen staging column. Visit FIVE named checkpoints in order: S1 (40,4), S2 (75,36), S3 (95,4), S4 (118,36), S5 (118,20). Read your start position from the observation; the chain is latched the same regardless. Within about 50 turns.",
    'Latched chain S1 → S2 → S3 → S4 → S5 (radius 5), within 4500 ticks.',
    'No jeeps left, or deadline (4501 ticks).',
    52, 4683)

# 2. proc-conditional-branch-action
P = 'proc-conditional-branch-action'
C = 'reasoning'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    '1× Construction Yard @(20,20) + 4× medium tank @(18-22,20-22) + 2× scout jeep forward-NORTH @(20-22,8)',
    'NORTH outpost: 1× barracks @(20,4) + 3× rifle infantry guard cluster @(20,6), guard bot; far-east sentinel fact @(120,20)',
    "Commander, this is an IF/ELSE runbook mission. The runbook reads: IF the enemy outpost is on the NORTH flank, send your tanks NORTH and raze the barracks; ELSE (outpost SOUTH) send them SOUTH. On this tier your two scout jeeps are already forward-north in sight of the outpost — read which side it is on from your observation and commit the four tanks to that flank. Keep your construction yard standing. Within about 60 turns.",
    'Scout sees the outpost (≥1 building discovered) AND the matching-flank barracks razed AND yard alive, within 5400 ticks.',
    'Yard lost, or deadline (5401 ticks).',
    66, 5943)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    '1× fact @(20,20) + 4× 2tnk + 2× scout jeep at base @(24-26,20)',
    'NORTH outpost (tent+3×e1 cluster at y=4-6), guard bot; sentinel fact @(120,20)',
    "Commander, same IF/ELSE runbook — IF the outpost is NORTH then attack NORTH, ELSE attack SOUTH — but the enemy flank is NOT given. Drive a jeep forward to discover the outpost, read which flank it is on, then send the four tanks to that flank and raze the barracks. Yard alive, within about 60 turns. Skipping the scout leaves the then-chain stuck at clause one.",
    'Scout sees the outpost AND the matching-flank barracks razed AND yard alive, within 5400 ticks.',
    'Yard lost, or deadline (5401 ticks).',
    66, 5943)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    '1× fact @(20,20) + 4× 2tnk + 2× scout jeep at base; outpost flank flips per seed (enemy-side sp0/sp1)',
    'Seeded outpost: sp0 NORTH @(20,4) or sp1 SOUTH @(20,36), guard bot; sentinel fact @(120,20) (duplicated)',
    "Commander, same conditional runbook, but the outpost may be on the NORTH OR the SOUTH flank — chosen by seed and not given. Drive a scout jeep to discover the live flank, then commit the four tanks to that flank and raze the barracks. A memorised always-NORTH (or always-SOUTH) opener strands the strike force on the empty lane on half the seeds. Within about 60 turns.",
    'Scout sees the outpost AND the matching-flank barracks razed AND yard alive, within 5400 ticks.',
    'Yard lost, or deadline (5401 ticks).',
    66, 5943)

# 3. proc-instruction-following-edge-case
P = 'proc-instruction-following-edge-case'
C = 'action'
T = 'move_units, attack_unit, observe, stop'
add(P, 'easy', C, M, SZ, 'fit', T, '[]', 'none',
    '3× jeep @(8,18-21) + 3× medium tank @(8-10,20) on the west edge',
    'No enemy; egress region at (90,20)',
    "Commander, this is a SELECTIVE-action drill. Move ALL THREE jeeps to the region around (90,20) within radius 6, before tick 4500. On this calibration tier the tanks may be ignored (no negative-instruction teeth yet). Stalling and missed deadline LOSE.",
    'Three jeeps inside the (90,20) disc within 4500 ticks.',
    'No units left, or deadline (4501 ticks).',
    55, 4953)
add(P, 'medium', C, M, SZ, 'fit', T, '[]', 'none',
    '3× jeep @(8,18-21) + 3× 2tnk @(8-10,20) on the west edge',
    'No enemy; egress region (90,20)',
    "Commander, SELECTIVE-action mission with both clauses live. Move ALL THREE jeeps to the disc at (90,20) within radius 6. Do NOT move the THREE tanks — they must remain inside the disc at (8,20). Issuing any move order to a tank breaks the tank-at-start clause as soon as a tank leaves the cluster and FAILS the run instantly. Within about 50 turns.",
    'Three jeeps in the (90,20) disc AND three tanks still in the (8,20) disc within 4500 ticks.',
    'A tank leaves the start disc, no units left, or deadline (4501 ticks).',
    55, 4953)
add(P, 'hard', C, M, SZ, 'fit', T, '[]', 'none',
    'Jeeps seed-spawned NW (sp0 @8,7-9) or SW (sp1 @8,31-33); 3× 2tnk anchored at (8-10,20) across BOTH spawn groups',
    'No enemy',
    "Commander, same SELECTIVE-action drill: three jeeps to (90,20), three tanks held at (8,20). The jeeps now spawn at the NW (y≈8) OR SW (y≈32) corner by seed; route them accordingly. Any move order on a tank is an instant fail. Within about 50 turns.",
    'Three jeeps in the (90,20) disc AND three tanks still in the (8,20) disc within 4500 ticks.',
    'A tank leaves the start disc, no units left, or deadline (4501 ticks).',
    55, 4953)

# 4. proc-no-attack-passive-only
P = 'proc-no-attack-passive-only'
C = 'action'
T = 'move_units, stop, observe'
FT = 'attack_unit, attack_move'
add(P, 'easy', C, M, SZ, 'wide', T, FT, 'none',
    '4× jeep @(8-9,18-22) on HoldFire',
    'ONE sentry post: 4× gun turret bracketing y=18-22 at x=55 (stance:3 AttackAnything)',
    "Commander, RULES-OF-ENGAGEMENT recon mission. You are a scout column with only move and stop — firing is FORBIDDEN. Reach the observation point at (110,20) within radius 6, losing no more than two of four jeeps, before tick 4500. One sentry post of four gun turrets covers the central lane (y=20). Detour NORTH (y≈8) or SOUTH (y≈32) past them, then drop in. Within about 50 turns.",
    'A jeep inside the (110,20) disc AND ≤2 lost AND no attack-tool calls, within 4500 ticks.',
    'Any attack-tool call, >2 lost, no units left, or deadline (4501 ticks).',
    55, 4953)
add(P, 'medium', C, M, SZ, 'wide', T, FT, 'none',
    '4× jeep @(8-9,18-22) on HoldFire',
    'TWO sentry posts: 4× gun turret each at x=45 and x=75 bracketing y=18-22',
    "Commander, same ROE recon — move and stop only, firing is forbidden. Two sentry posts of four turrets each now hold the central lane at x=45 and x=75 — a column driven straight through dies. Detour NORTH or SOUTH past both. Reach (110,20) inside radius 6, lose at most one of four. Within about 50 turns.",
    'A jeep in the (110,20) disc AND ≤1 lost AND no attack-tool calls, within 4500 ticks.',
    'Any attack-tool call, >1 lost, no units left, or deadline (4501 ticks).',
    55, 4953)
add(P, 'hard', C, M, SZ, 'wide', T, FT, 'none',
    '4× jeep @(8-9,18-22) on HoldFire',
    'THREE sentry posts at x=40/65/90 (12× gun turrets), seed-randomised anchor offset ±2',
    "Commander, same ROE recon with three sentry posts along the central lane and randomised anchor offsets — the exact turret cells shift between runs, so memorised cells will not generalise. Reach (110,20) inside radius 6 with ALL FOUR jeeps alive. Any attack call fails the run. Within about 50 turns.",
    'A jeep in the (110,20) disc AND zero losses AND no attack-tool calls, within 4500 ticks.',
    'Any attack-tool call, any loss, no units left, or deadline (4501 ticks).',
    55, 4953)

# 5. proc-only-build-no-combat
P = 'proc-only-build-no-combat'
C = 'action'
T = 'build, place_building, move_units, stop, observe'
FT = 'attack_unit, attack_move'
add(P, 'easy', C, M, SZ, 'wide', T, FT, 'none',
    'Pre-placed base: fact @(10,18) + tent @(14,18) + powr @(10,21) + proc @(16,21); starting_cash $2600',
    'Far-east passive squad: 3× rifle infantry @(100,30-31) on HoldFire; sentinel fact @(120,20)',
    "Commander, you are a BUILDER role. Allowed tools: build, place_building, move_units, observe, stop. attack_unit and attack_move are FORBIDDEN — a single forbidden call fails the run. Your base is seeded with a Construction Yard, Barracks, Power Plant, and Refinery. Stand up at least two more buildings ending in a War Factory for SIX total, within about 60 turns. A distant passive rifle squad sits off the minimap edge but is not your concern.",
    'Building total ≥6 AND has War Factory AND yard alive, within 5400 ticks.',
    'Any attack call, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'medium', C, M, SZ, 'wide', T, FT, 'none',
    'Same pre-placed base; agent spotter e1 @(25,20) HoldFire extends vision east; starting_cash $2400',
    'Visible passive squad: 3× rifle infantry @(28,19-21) HoldFire (no damage but visible temptation); sentinel fact @(120,20)',
    "Commander, same BUILDER role and tool ban. Your base is seeded (yard, barracks, power, refinery). Build at least two more buildings to reach six total, ending with a War Factory. A visible enemy rifle squad sits east of your base on hold-fire — ignore it, your role is build-only. One attack call fails the run. Within about 60 turns.",
    'Building total ≥6 AND has War Factory AND yard alive, within 5400 ticks.',
    'Any attack call, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'hard', C, M, SZ, 'wide', T, FT, 'none',
    'Same pre-placed base; agent spotter @(20,20) HoldFire; starting_cash $2900',
    'Active hunt squad: 3× rifle infantry @(40,18-22) stance:2 under hunt bot — chips structures but does not raze the yard',
    "Commander, same BUILDER role. A hostile rifle squad now approaches from the east and will chip your structures. Stand up enough additional buildings to reach SEVEN total, ending with a War Factory. You may move your defenders but you may NOT order an attack — one attack call fails the run. Within about 60 turns.",
    'Building total ≥7 AND has War Factory AND yard alive, within 5400 ticks.',
    'Any attack call, yard destroyed, or deadline (5401 ticks).',
    60, 5403)

# 6. proc-only-defend-no-attack
P = 'proc-only-defend-no-attack'
C = 'action'
T = 'move_units, set_stance, stop, observe'
FT = 'attack_unit, attack_move'
add(P, 'action', C, M, SZ, 'fit', T, FT, 'none',
    'PLACEHOLDER (replaced below)', '', '', '', '', 0, 0)
R.pop()
add(P, 'easy', C, M, SZ, 'fit', T, FT, 'none',
    '1× fact @(16,20) + 5× rifle infantry defenders clustered @(13-14,18-22) on HoldFire (stance:0)',
    '3× rifle infantry patrol probes at central anchor (22,20), patrol bot (oscillates 14..30)',
    "Commander, DEFENSIVE ROE. Hold your yard at (16,20) against three raiders patrolling the central corridor. Your five rifles are on HoldFire — only set_stance and move_units are available, and attack_unit/attack_move are FORBIDDEN; one call is an instant fail. Lift the stance, position the defenders, and let the engine's auto-fire score the kills. Two kills, yard alive, within about 50 turns.",
    'Yard alive AND ≥2 kills (from defender auto-fire) AND no attack calls, within 4500 ticks.',
    'Any attack call, yard destroyed, or deadline (4501 ticks).',
    55, 4953)
add(P, 'medium', C, M, SZ, 'fit', T, FT, 'none',
    '1× fact @(16,20) + 5× rifle defenders on HoldFire',
    '4× rifle infantry patrol probes @(22,20), patrol bot',
    "Commander, same defensive ROE with four patrol raiders on the central corridor. Use set_stance and move_units to position and arm the five rifles; attack calls fail the run instantly. Two kills, yard alive, within about 50 turns.",
    'Yard alive AND ≥2 kills AND no attack calls, within 4500 ticks.',
    'Any attack call, yard destroyed, or deadline (4501 ticks).',
    55, 4953)
add(P, 'hard', C, M, SZ, 'fit', T, FT, 'none',
    '1× fact @(16,20) duplicated across sp0/sp1; 5× rifle defenders seed-spawned NORTH y=17-19 (sp0) or SOUTH y=21-23 (sp1)',
    '5× rifle infantry patrol probes @(22,20), patrol bot',
    "Commander, same defensive ROE with FIVE raiders. Your defenders stage at the NORTH (y≈18) or SOUTH (y≈22) row by seed, both covering the yard column. Position and arm with set_stance/move_units only — attack calls fail instantly. Three kills, yard alive, within about 50 turns.",
    'Yard alive AND ≥3 kills AND no attack calls, within 4500 ticks.',
    'Any attack call, yard destroyed, or deadline (4501 ticks).',
    55, 4953)

# 7. proc-ordered-action-strict
P = 'proc-ordered-action-strict'
C = 'action'
T = 'build, place_building, observe'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'Pre-placed agent base: fact @(10,18) + tent @(14,18) + powr @(10,22); starting_cash $3000',
    'Sentinel enemy fact @(115,20)',
    "Commander, STRICT ORDERED PROCEDURE. Place pillboxes (stationary defensive turrets) at exact regions IN ORDER: A at (30,20) FIRST, then B at (90,20). Placing B before A — a pillbox at B while none exists at A — is an immediate loss; the procedure is unrecoverable. Within about 29 turns.",
    'Then-chain pbox-at-A → pbox-at-B (radius 5), within 2600 ticks.',
    'Out-of-order placement (B before A), yard destroyed, or deadline (2601 ticks).',
    30, 2703)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'Same pre-placed agent base; starting_cash $3000',
    'Sentinel enemy fact @(115,20)',
    "Commander, same STRICT ORDERED procedure. Place pillboxes at three regions IN ORDER: A (30,20), then B (60,20), then C (90,20). Any later step placed before its predecessor is an immediate loss. Within about 37 turns.",
    'Then-chain pbox A → B → C (radius 5), within 3300 ticks.',
    'Out-of-order placement, yard destroyed, or deadline (3301 ticks).',
    38, 3423)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'Same pre-placed agent base; starting_cash $4000',
    'Sentinel enemy fact @(115,20)',
    "Commander, same STRICT ORDERED procedure with FOUR ordered placements: A (30,20), B (50,20), C (70,20), D (90,20). Any out-of-order placement is an instant loss; no spare pieces (more than seven total buildings also loses — you cannot brute-force a recovery). Within about 47 turns.",
    'Then-chain A → B → C → D (radius 5) AND ≤7 total buildings, within 4200 ticks.',
    'Out-of-order placement, >7 total buildings, yard destroyed, or deadline (4201 ticks).',
    48, 4323)

# 8. proc-strict-toolban-fidelity
P = 'proc-strict-toolban-fidelity'
C = 'action'
T = 'move_units, build, place_building, observe, stop'
FT = 'attack_unit, attack_move, set_stance, harvest, repair, sell, set_rally_point'
add(P, 'easy', C, M, SZ, 'wide', T, FT, 'none',
    '1× fact @(10,20) + 4× jeep @(8-9,19-21) on HoldFire',
    'Sentinel enemy fact @(120,20); no combatants',
    "Commander, STRICT ALLOWLIST fidelity. Move at least three of your four jeeps to the egress region around (90,20) within radius 6, before tick 5400. Tool allowlist: move_units, build, place_building, observe, stop. FORBIDDEN: attack_unit, attack_move, set_stance, harvest, repair, sell, set_rally_point — one call to any of these fails the run instantly. Yard at (10,20) must survive. Within about 60 turns.",
    'Three jeeps in the (90,20) disc AND yard alive AND no forbidden-tool calls, within 5400 ticks.',
    'Any forbidden-tool call, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'medium', C, M, SZ, 'wide', T, FT, 'none',
    '1× fact @(10,20) + 4× jeep @(8-9,19-21) on HoldFire',
    'Sentinel enemy fact @(120,20); no combatants',
    "Commander, same allowlist and forbidden list on a TIGHTER clock — three jeeps to (90,20) inside radius 6, yard alive, no forbidden tool calls. Stalling or a slow convoy misses the deadline. Within about 40 turns.",
    'Three jeeps in the (90,20) disc AND yard alive AND no forbidden-tool calls, within 3600 ticks.',
    'Any forbidden-tool call, yard destroyed, or deadline (3601 ticks).',
    41, 3693)
add(P, 'hard', C, M, SZ, 'wide', T, FT, 'none',
    '1× fact @(10,20) duplicated across sp0/sp1; 4× jeep seed-spawned NW y≈8 (sp0) or SW y≈32 (sp1), HoldFire',
    'Sentinel enemy fact @(120,20); no combatants',
    "Commander, same strict allowlist and clock. The jeeps may spawn NW (y≈8) or SW (y≈32) by seed — read your start and route accordingly. Three jeeps in the (90,20) disc, yard alive, no forbidden-tool calls. Within about 40 turns.",
    'Three jeeps in the (90,20) disc AND yard alive AND no forbidden-tool calls, within 3600 ticks.',
    'Any forbidden-tool call, yard destroyed, or deadline (3601 ticks).',
    41, 3693)

# 9. proc-tool-use-multi-distractor
P = 'proc-tool-use-multi-distractor'
C = 'action'
T = 'move_units, attack_unit, attack_move, build, place_building, harvest, set_stance, set_rally_point, observe'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    '4× jeep @(8-9,19-21) on HoldFire + agent base (fact @14,20 + tent @18,20 + powr @14,24); starting_cash $2000',
    'Off-path garrison: 1× rifle infantry @(45,8) on HoldFire; sentinel fact @(118,20)',
    "Commander, CLUTTERED-TOOL discipline. Get a jeep to the observation point at (110,20) within radius 6 with zero losses, before tick 2000. You have a nine-tool API — most of it irrelevant. Use only move_units and observe; spending turns on attack, build, harvest, or stance calls runs the clock out. A weak off-path rifleman to the north and the western base are distractions. Within about 23 turns.",
    'A jeep in the (110,20) disc AND zero losses, within 2000 ticks.',
    'Any loss, no units left, or deadline (2001 ticks).',
    25, 2253)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    '4× jeep on HoldFire + agent base; starting_cash $3000',
    'Off-path garrison: 3× rifle infantry @(45,8-9) HoldFire; sentinel fact @(118,20)',
    "Commander, same cluttered nine-tool API on a TIGHTER clock. Get a jeep to (110,20) inside radius 6, zero losses, within about 17 turns. A three-rifle off-path garrison and the west base are tempting but pure distractions. Use only move_units and observe.",
    'A jeep in the (110,20) disc AND zero losses, within 1500 ticks.',
    'Any loss, no units left, or deadline (1501 ticks).',
    22, 1983)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    '4× jeep seed-spawned NORTH y=4-6 (sp0) or SOUTH y=34-36 (sp1), HoldFire; base duplicated across both groups; starting_cash $3000',
    'Off-path garrison @(45,8-9) HoldFire AND central corridor sentry stack @(60,18-22) HoldFire (briefing says stance:3 but YAML has stance:0); sentinel fact @(118,20)',
    "Commander, same cluttered-tool task on the tightest clock. The jeeps spawn NORTH (y≈5) or SOUTH (y≈35) by seed. Two visible distractor groups — an off-path north garrison and a central corridor sentry stack — will pull an undisciplined column off course. Use move_units and observe only; route along your spawn lane and drop in at the end. Within about 17 turns.",
    'A jeep in the (110,20) disc AND zero losses, within 1500 ticks.',
    'Any loss, no units left, or deadline (1501 ticks).',
    20, 1803)

# 10. proc-tool-use-with-distractor
P = 'proc-tool-use-with-distractor'
C = 'action'
T = 'move_units, attack_unit, build, place_building, observe'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    '1× fact @(4,20) + 3× jeep @(8,19-21) on ReturnFire (stance:1); starting_cash $1200',
    'Single passive sentry: 1× rifle infantry @(40,32) HoldFire; sentinel fact @(120,20)',
    "Commander, DISTRACTOR-TOOL discipline. Three jeeps at the west lane mouth. Drive one to (110,20) inside radius 6 with zero losses. The tools list shows attack, build, place_building — none of them are needed; spending turns on them runs out the clock. There is a passive sentry well off-lane and a far-east enemy marker — neither requires combat. Within about 50 turns.",
    'A jeep in the (110,20) disc AND zero losses, within 4500 ticks.',
    'Any loss, no units left, or deadline (4501 ticks).',
    55, 4953)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    '1× fact @(4,20) + 3× jeep on ReturnFire; starting_cash $1200',
    'Light patrol: 1× rifle infantry @(60,20) stance:2, patrol bot (oscillates ±8); sentinel fact @(120,20)',
    "Commander, same distractor task with a single patrol rifleman now oscillating across the central corridor near (60,20). Engaging it or queuing buildings instead of moving runs out the clock. Reach (110,20) inside radius 6, zero losses, within about 50 turns.",
    'A jeep in the (110,20) disc AND zero losses, within 4500 ticks.',
    'Any loss, no units left, or deadline (4501 ticks).',
    55, 4953)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    '3× jeep seed-spawned NORTH y=7-9 (sp0) or SOUTH y=31-33 (sp1), ReturnFire; fact at the spawn-matched latitude',
    'Central patrol + flank patrol probes (one at y=12, one at y=28), patrol bot; sentinel fact @(120,20)',
    "Commander, same distractor task on a NORTH (y≈8) or SOUTH (y≈32) spawn lane by seed. Two patrol probes (central corridor and a flank band) are active. The only win path is movement on the spawn lane; building or attacking instead of moving runs out the clock. Within about 50 turns.",
    'A jeep in the (110,20) disc AND zero losses, within 4500 ticks.',
    'Any loss, no units left, or deadline (4501 ticks).',
    55, 4953)

# ──────────────────────────────────────────────────────────────────
# strict-* packs (3)
# ──────────────────────────────────────────────────────────────────

# 11. strict-production-bom
P = 'strict-production-bom'
C = 'action'
T = 'build, place_building, deploy, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'fact @(10,18) + powr @(14,18); starting_cash $2200; Soviet faction',
    'Single rifle infantry @(110,34) on Defend',
    "Commander, EXACT BILL OF MATERIALS. Build EXACTLY three rifles AND EXACTLY one grenadier — both need a barracks first. The budget covers the barracks plus the precise mix. A fourth rifle, a second grenadier, or any extra structure breaks the spec. Deliver within about 30 turns.",
    'Exactly 3× rifle (e1) AND exactly 1× grenadier (e2), within 2700 ticks.',
    '≥4 rifles, ≥2 grenadiers, or deadline (2701 ticks).',
    40, 3603)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'fact @(10,18) + powr @(14,18); starting_cash $3000; Soviet',
    'Single rifle @(110,34) on Defend',
    "Commander, same BOM discipline at a different mix: build EXACTLY three rifles AND EXACTLY two rocket soldiers. The rocket soldiers share the barracks tech but cost more — the budget forces correct allocation, no waste. Over- or under-producing either type, or missing the deadline, is a loss. Within about 33 turns.",
    'Exactly 3× rifle (e1) AND exactly 2× rocket soldier (e3), within 3000 ticks.',
    '≥4 rifles, ≥3 rockets, or deadline (3001 ticks).',
    45, 4053)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'fact @(10,18) + powr @(14,18); starting_cash $7200; Soviet',
    'Single rifle @(110,34) on Defend',
    "Commander, full BOM with tech depth: EXACTLY three rifles AND EXACTLY two rocket soldiers, PLUS a working Tesla coil. The Soviet tech path is refinery → war factory → Tesla coil; Tesla draws heavy power so extra power plants are required to keep the surplus non-negative. The tight budget leaves no room for overproduction. Within about 70 turns.",
    'Exactly 3× rifle AND exactly 2× rocket AND ≥1 Tesla coil AND power surplus ≥0, within 6300 ticks.',
    '≥4 rifles, ≥3 rockets, ≥2 Tesla coils, or deadline (6301 ticks).',
    75, 6753)

# 12. strict-sequence
P = 'strict-sequence'
C = 'action'
T = 'move_units, stop'
add(P, 'easy', C, M, SZ, 'fit', T, '[]', 'none',
    '1× jeep @(6,8) on HoldFire',
    'No enemy; three staging points at the listed cells',
    "Commander, STRICT ORDERED EXECUTION. Visit three staging points in EXACTLY this order with only move_units and stop: S1 (40,30), S2 (95,10), S3 (118,33). A later point reached early does not count or skip ahead. Idling runs out the clock. Within about 29 turns.",
    'Latched chain S1 → S2 → S3 (radius 5), within 2600 ticks.',
    'No jeep left, or deadline (2601 ticks).',
    34, 3063)
add(P, 'medium', C, M, SZ, 'fit', T, '[]', 'none',
    '1× jeep @(6,8) on HoldFire',
    'No enemy; four staging points',
    "Commander, same strict-order procedure with FOUR staging points: S1 (40,30), S2 (95,10), S3 (40,33), S4 (118,20). A skipped point is unrecoverable. Within about 38 turns.",
    'Latched chain S1 → S2 → S3 → S4 (radius 5), within 3400 ticks.',
    'No jeep left, or deadline (3401 ticks).',
    44, 3963)
add(P, 'hard', C, M, SZ, 'fit', T, '[]', 'none',
    '1× jeep seed-spawned NW (sp0 @6,8) or SW (sp1 @6,30), HoldFire',
    'No enemy; same four staging points',
    "Commander, same four staging points in order — S1 (40,30), S2 (95,10), S3 (40,33), S4 (118,20) — from a seed-chosen staging corner. Order is latched. Within about 42 turns.",
    'Latched chain S1 → S2 → S3 → S4 (radius 5), within 3700 ticks.',
    'No jeep left, or deadline (3701 ticks).',
    46, 4143)

# 13. strict-toolban-fidelity-under-pressure
P = 'strict-toolban-fidelity-under-pressure'
C = 'action'
T = 'move_units, stop, observe'
FT = 'attack_unit, attack_move'
add(P, 'easy', C, M, SZ, 'wide', T, FT, 'none',
    '3× rifle infantry @(8,19-21) on HoldFire (stance:0)',
    '1× rifle infantry @(60,20) patrol bot (anchor 60,20, oscillates 52..68); sentinel fact @(120,20)',
    "Commander, STRICT TOOL-BAN under pressure. Move your three rifles to the egress region around (110,20) within radius 6 before tick 4500. A single enemy rifleman patrols the central corridor between (52,20) and (68,20). You may NOT call attack_unit or attack_move — one call is an instant fail. One unit in the egress disc, at most one loss, within about 50 turns.",
    'A unit in (110,20) disc AND ≤1 lost AND no attack calls, within 4500 ticks.',
    'Any attack call, >1 lost, or deadline (4501 ticks).',
    55, 4953)
add(P, 'medium', C, M, SZ, 'wide', T, FT, 'none',
    '3× rifle infantry @(8,19-21) HoldFire',
    '2× rifle infantry @(60,20) patrol bot (stacked); sentinel fact @(120,20)',
    "Commander, same egress task and tool ban with TWO rifles patrolling stacked on the central corridor near (60,20). Same allowlist; one attack call is an instant fail. One unit in the disc, at most one loss, within about 50 turns.",
    'A unit in (110,20) disc AND ≤1 lost AND no attack calls, within 4500 ticks.',
    'Any attack call, >1 lost, or deadline (4501 ticks).',
    55, 4953)
add(P, 'hard', C, M, SZ, 'wide', T, FT, 'none',
    '3× rifle infantry @(8,19-21) HoldFire',
    '2× rifle @(60,20) AND 2× rifle @(60,10) patrol bots; sentinel fact @(120,20)',
    "Commander, same egress task and tool ban with TWO patrol vectors now active — a central arc near (60,20) and a northern arc near (60,10). ZERO losses allowed; one attack call is an instant fail. One unit in the (110,20) disc, within about 50 turns.",
    'A unit in (110,20) disc AND zero losses AND no attack calls, within 4500 ticks.',
    'Any attack call, any loss, or deadline (4501 ticks).',
    55, 4953)

# ──────────────────────────────────────────────────────────────────
# maint-* packs (2)
# ──────────────────────────────────────────────────────────────────

# 14. maint-repair-priority-order
P = 'maint-repair-priority-order'
C = 'reasoning'
T = 'observe, repair'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'spawn@720:e2×9 destroy@820:enemy-region',
    'Pre-damaged base: proc (40%) @(20,12), weap (55%) @(20,22), fact (50%) @(20,32), pbox (20%) @(44,16), fix (35%) @(44,28); starting_cash $4000',
    'Sentinel fact @(120,20); strike (3 grenadier squads, 3 each) spawns at tick 720 adjacent to the three critical buildings, removed at tick 820',
    "Commander, PREVENTIVE-MAINTENANCE triage. You inherit a quiet production base — a refinery (~40% HP), war factory (~55%), construction yard (~50%), pillbox (~20%), and service depot (~35%). A grenadier strike will hit the refinery, war factory, and yard around turn 9. The worn pillbox looks the most urgent but is off the strike path. Toggle repair on the three critical buildings and keep it on through the strike. Within about 17 turns.",
    'Refinery, war factory, AND yard still standing after tick 900 AND within 1500 ticks.',
    'Refinery destroyed, or deadline (1501 ticks).',
    22, 1983)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'spawn@720:e2×12 destroy@820:enemy-region',
    'Same pre-damaged base; starting_cash $4000',
    'Heavier strike (3 squads of 4 grenadiers each)',
    "Commander, same damaged base and off-path pillbox decoy, but the grenadier strike is heavier and the clock tighter. Repair the refinery, war factory, and yard to full before turn 9 and keep repair on through the strike. Within about 15 turns.",
    'Refinery, war factory, AND yard still standing after tick 900 AND within 1300 ticks.',
    'Refinery destroyed, or deadline (1301 ticks).',
    20, 1803)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'spawn@720:e2×24 destroy@820:wide-region',
    'Pre-damaged base seed-spawned at WEST x=20 (sp0) or EAST x=70 (sp1) column; starting_cash $4000',
    'Both base columns receive an adjacent grenadier strike at tick 720 (12 grenadiers per column); destroy_actors at tick 820 removes them',
    "Commander, same damaged base and heavy strike, but the base spawns from a seed-chosen column — a memorised opening fails. Read the actual building cells from your observation, toggle repair on the refinery, war factory, and yard, keep it on through the strike. Within about 15 turns.",
    'Refinery, war factory, AND yard still standing after tick 900 AND within 1300 ticks.',
    'Refinery destroyed, or deadline (1301 ticks).',
    20, 1803)

# 15. maint-sell-and-recoup-cash
P = 'maint-sell-and-recoup-cash'
C = 'reasoning'
T = 'observe, sell, build, place_building'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'Core base: fact (4,4) + proc (4,9) + 2× powr + fix (8,9); obsolete cluster: 2× pbox + tsla + dome (refund $1650); starting_cash $3500; no income (proc has no reachable ore patch)',
    'Sentinel fact @(122,20); no combatants',
    "Commander, CAPITAL REALLOCATION. Your NW base holds a yard, refinery, two power plants, and a service depot — plus an over-built static-defence cluster (two pillboxes, a Tesla coil, a radar dome) the new mission no longer needs. You have $3500 cash and no income source. The objective has shifted to offence: build a War Factory and field three medium tanks. Cash alone cannot fund the war factory ($2000) and the three-tank batch (~$2400). Sell the obsolete defences (each refunds 50% of its build cost), then build. Yard alive, within about 87 turns.",
    'Has War Factory AND ≥3 medium tanks AND yard alive, within 7800 ticks.',
    'Yard destroyed, or deadline (7801 ticks).',
    90, 8103)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'Same core base; obsolete cluster: 3× pbox + tsla + hbox (refund $1650); starting_cash $3000',
    'Sentinel fact @(122,20); no combatants',
    "Commander, same NW base with a different obsolete cluster — three pillboxes, a Tesla coil, a heavy pillbox. $3000 cash and no income. Sell to recoup capital, build the war factory, field three medium tanks. Yard alive, within about 87 turns.",
    'Has War Factory AND ≥3 medium tanks AND yard alive, within 7800 ticks.',
    'Yard destroyed, or deadline (7801 ticks).',
    90, 8103)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'Core base + obsolete cluster seed-spawned NORTH y=4 (sp0) or SOUTH y=34 (sp1); obsolete: 3× pbox + tsla + dome (refund $1850); starting_cash $2800',
    'Sentinel fact @(122,20); no combatants',
    "Commander, same divestment task but the base spawns NORTH or SOUTH by seed. Obsolete cluster is three pillboxes plus Tesla and radar dome. $2800 cash and no income. Sell to recoup capital, build the war factory, field three medium tanks. Yard alive, within about 87 turns.",
    'Has War Factory AND ≥3 medium tanks AND yard alive, within 7800 ticks.',
    'Yard destroyed, or deadline (7801 ticks).',
    90, 8103)

# ──────────────────────────────────────────────────────────────────
# rob-* packs (8)
# ──────────────────────────────────────────────────────────────────

# 16. rob-cash-depletion-recovery
P = 'rob-cash-depletion-recovery'
C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'fact + powr + proc + 1× harv + 2× ore patches + 2× 3tnk defenders @(18,17-19) stance:2; starting_cash $1400',
    'Light strike: 1× heavy tank (4tnk) adjacent to the proc, hunt bot; sentinel fact @(120,20)',
    "Commander, ECONOMY RECOVERY. You start with a Construction Yard, Power, Refinery, one Harvester, two near ore patches, two heavy-tank defenders, and $1400. A lone enemy heavy tank is staged beside the refinery — your defenders intercept and the refinery survives. Just commit the harvester to a patch cell. Reach economy value 2500 with the refinery and yard alive, within about 70 turns.",
    'Economy value ≥2500 AND refinery alive AND yard alive, within 6300 ticks.',
    'Yard destroyed, or deadline (6301 ticks).',
    70, 6303)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'fact + powr + proc + 1× harv + 2× patches + 3× 3tnk defenders @(18,17-19) stance:2; starting_cash $1400',
    'Medium strike: 2× heavy tank @(14,17/19) adjacent to proc, hunt — the proc is razed around tick 270 but the strike is then wiped; sentinel fact @(120,20)',
    "Commander, ECONOMY RECOVERY after a strike. Same base and $1400 reserve, but TWO enemy heavies attack the refinery and overwhelm the defenders — the refinery falls around tick 270 and the harvest loop dies. Queue another refinery ($1400 — the reserve covers it exactly), place it beside the yard, then re-issue harvest on the surviving worker. Economy value 2500, refinery and yard alive, within about 70 turns.",
    'Economy value ≥2500 AND refinery alive AND yard alive, within 6300 ticks.',
    'Yard destroyed, or deadline (6301 ticks).',
    70, 6303)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'fact + powr + proc + tent + 1× harv + 2× patches + 5× 3tnk defenders; starting_cash $1400',
    'Heavy strike: 4× heavy tank around the proc + tent (tent dies ~tick 183, proc ~tick 273), hunt; sentinel fact @(120,20). Spawn-point varied (NORTH y=14 / SOUTH y=26)',
    "Commander, same economy recovery with a heavier strike. Four enemy heavies attack the proc and the barracks — both fall early in the episode. The indivisible $1400 reserve only covers ONE structure — the refinery (the income channel) is the load-bearing rebuild; the barracks is not on the win path. Reach economy value 2500 with refinery and yard alive, within about 70 turns.",
    'Economy value ≥2500 AND refinery alive AND yard alive, within 6300 ticks.',
    'Yard destroyed, or deadline (6301 ticks).',
    70, 6303)

# 17. rob-deadline-shortened-midway
P = 'rob-deadline-shortened-midway'
C = 'reasoning'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'fit', T, '[]', 'shorten@1000:max_ticks=2000',
    '4× medium tank @(15-17,19-21) on HoldFire + fact @(10,20); starting_cash $1000',
    'Enemy fact (the objective) @(100,20); sentinel none — objective IS the only enemy',
    "Commander, SCHEDULE COMPRESSION. Four medium tanks at the WEST base, enemy yard far EAST. DEADLINE WARNING: at tick 1000 the time budget gets CUT to tick 2000. Commit all four tanks east at full speed from turn 1 and switch to attack the enemy yard the moment it appears. Stalling or a leisurely opening loses the compressed budget.",
    'Enemy fact razed AND own yard alive, within 2000 ticks.',
    'Yard destroyed, or deadline (2001 ticks).',
    60, 5403)
add(P, 'medium', C, M, SZ, 'fit', T, '[]', 'shorten@1000:max_ticks=2000',
    '4× medium tank on HoldFire + own fact; starting_cash $1000',
    'Enemy fact deeper east at (108,20) — 8 cells further than easy',
    "Commander, same schedule compression with the enemy yard now 8 cells deeper east. The tick-1000 deadline cut still clamps the budget to 2000 — the slack shrinks to one decision turn. Commit four tanks east immediately and switch to attack the yard the instant it appears. Within about 22 turns.",
    'Enemy fact razed AND own yard alive, within 2000 ticks.',
    'Yard destroyed, or deadline (2001 ticks).',
    60, 5403)
add(P, 'hard', C, M, SZ, 'fit', T, '[]', 'shorten@1000:max_ticks=2050',
    '4× medium tank seed-varied staging + own fact; starting_cash $1000',
    'Enemy fact at the deep east; seed varies staging cluster',
    "Commander, same schedule compression with a seed-varied staging cluster — read your start cells from the observation. The tick-1000 deadline cut clamps the budget to 2050; an immediate full-speed commit makes it, any dawdle does not. Within about 23 turns.",
    'Enemy fact razed AND own yard alive, within 2050 ticks.',
    'Yard destroyed, or deadline (2051 ticks).',
    60, 5403)

# 18. rob-multiple-simultaneous-pressures
P = 'rob-multiple-simultaneous-pressures'
C = 'reasoning'
T = 'observe, build, place_building, harvest, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'fact @(10,18) + proc @(8,18) + powr @(8,20) + 2× harv on near patch + 2× mines + 4× 1tnk defenders ringing the patch; starting_cash $2000 (exact weap cost)',
    'Base-hunt squad: 3× rifle infantry @(40,4) stance:3 hunt — charges the fact. Patch-raid: 4× 1tnk @(60,16/22 + 63,16/22) hunt — targets nearest harv. Sentinel fact @(120,4)',
    "Commander, COMPOUND INCIDENT. You hold a base with yard, refinery, power, two harvesters on the near patch, and four light-tank defenders. $2000 — exactly the cost of a war factory. THREE pressures hit at once: a three-rifle hunt squad charging the base from the north, four enemy raider tanks targeting your harvesters from the patch lane, AND a tech bar (war factory required by tick 5400). Command the defenders actively, queue the war factory, keep at least one harvester alive and harvesting. Yard standing, economy value 1000 by tick 5400.",
    'Has War Factory AND has Refinery AND ≥1 harvester AND yard alive AND economy value ≥1000, within 5400 ticks.',
    'Yard destroyed, no harvester, or deadline (5401 ticks).',
    61, 5493)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'Same base; starting_cash $2000',
    'Base-hunt: 4× rifle @(40,4) hunt (+1 vs easy). Patch-raid: 5× 1tnk on the patch lane mouth (+1 vs easy)',
    "Commander, same compound incident scaled up: four-rifle hunt squad on the north flank, FIVE raider tanks at the patch lane mouth, war factory still required by tick 5400. Command the defender ring actively, fund the war factory, keep at least one harvester alive. Same yard and economy-value bars. Within about 60 turns.",
    'Has War Factory AND has Refinery AND ≥1 harvester AND yard alive AND economy value ≥1000, within 5400 ticks.',
    'Yard destroyed, no harvester, or deadline (5401 ticks).',
    61, 5493)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'Same base; starting_cash $2000',
    'Base-hunt: 5× rifle (+1 vs medium). Patch-raid: 5× 1tnk + spawn variation; tighter clock 4500',
    "Commander, same compound incident on the tightest clock — five-rifle hunt squad and a five-tank patch raid, war factory required by tick 4500. The harvester loop must run THROUGH the raid (residual economy value 1000 is required AFTER the weap purchase). Yard standing, within about 50 turns.",
    'Has War Factory AND has Refinery AND ≥1 harvester AND yard alive AND economy value ≥1000, within 4500 ticks.',
    'Yard destroyed, no harvester, or deadline (4501 ticks).',
    51, 4593)

# 19. rob-objective-change-midway
P = 'rob-objective-change-midway'
C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop_units'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    '5× medium tank @(10,18-22) on ReturnFire + own fact @(6,20); starting_cash $500',
    'Region A: enemy proc @(55,8) + 1× rifle picket. Region B: enemy fact @(100,30) + 2× rifle guard. Sentinel fact @(125,4)',
    "Commander, ORDERED OBJECTIVE PIVOT. Five medium tanks at the west edge. PHASE 1: move three tanks into the forward staging region around (55,8) in the north. Once latched, PHASE 2: drive south-east and raze the enemy yard at (100,30) (two-rifle guard). Both phases IN ORDER. Beelining straight to the yard never registers phase 1; stopping at the staging area never satisfies phase 2. Within about 60 turns.",
    'Then-chain phase-1-stage → phase-2-raze AND own yard alive, within 5400 ticks.',
    'Yard destroyed, or deadline (5401 ticks).',
    65, 5853)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    '5× medium tank + own fact; starting_cash $500',
    'Region A: proc @(55,12) (laterally offset from base→B line) + 1× e1. Region B: fact @(105,34) + 2× e1. Sentinel fact @(125,4)',
    "Commander, same ordered pivot with regions A and B LATERALLY OFFSET. PHASE 1: stage three tanks around (55,12) north of centre. PHASE 2: drive south-east and raze the yard at (105,34). A straight drive to B no longer passes through A; the pivot is a real direction change. Within about 60 turns.",
    'Then-chain phase-1-stage → phase-2-raze AND own yard alive, within 5400 ticks.',
    'Yard destroyed, or deadline (5401 ticks).',
    65, 5853)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    '5× medium tank + own fact; starting_cash $500',
    'Region A: proc @(55,20) + e1 picket. Region B: fact @(100,30) + heavier guard. Sentinel fact @(125,4)',
    "Commander, same ordered pivot. PHASE 1: stage three tanks around (55,20). PHASE 2: drive south-east and raze the yard at (100,30) past its heavier guard. The phase-2 region remains the same, the staging shifts to centre. Within about 60 turns.",
    'Then-chain phase-1-stage → phase-2-raze AND own yard alive, within 5400 ticks.',
    'Yard destroyed, or deadline (5401 ticks).',
    65, 5853)

# 20. rob-objective-shift-with-or-clause
P = 'rob-objective-shift-with-or-clause'
C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop_units'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    '5× medium tank @(10,18-22) ReturnFire; starting_cash $500',
    'Yard A @(60,19) + 6× rocket soldiers (feasible-but-not-easy). Yard B @(100,30) + 1× rifle (easier). Sentinel fact @(125,4)',
    "Commander, FEASIBLE-PATH PIVOT. Five medium tanks at the western edge. Two enemy yards stand east: yard A at (60,20) guarded by six anti-tank rocket soldiers, and yard B at (100,30) guarded by one rifleman. Razing EITHER wins. A pure commit on A bleeds past the loss cap; a blind beeline to B drives through A's rocket arc. Scout A, recognise it's the harder path, pivot to B. Three losses max, within about 60 turns.",
    'EITHER yard razed AND ≤3 lost AND ≥1 unit alive, within 5400 ticks.',
    '>3 lost, no units, or deadline (5401 ticks).',
    65, 5853)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    '5× medium tank + same start; starting_cash $500',
    'Yard A @(60,19) + 8× rocket soldiers + 1× heavy tank (infeasible). Yard B @(100,30) + 2× rifle (feasible)',
    "Commander, same OR-pivot. Yard A is now at full guard: an 8-stack of rocket soldiers backed by a heavy tank — a pure-A commit definitively loses the cap. Yard B remains lightly guarded. Recognise A is too hot from a brief approach, detour south of its envelope, raze B. Within about 60 turns.",
    'EITHER yard razed AND ≤3 lost AND ≥1 unit alive, within 5400 ticks.',
    '>3 lost, no units, or deadline (5401 ticks).',
    65, 5853)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    '5× medium tank + same start; starting_cash $500',
    'Yard A heavily guarded; yard B mid-guard; loss cap tightens to 2',
    "Commander, same OR-pivot on a tighter loss cap of two. Yard A is the hot target, yard B the cooler one but no longer trivial. Two losses max, within about 60 turns.",
    'EITHER yard razed AND ≤2 lost AND ≥1 unit alive, within 5400 ticks.',
    '>2 lost, no units, or deadline (5401 ticks).',
    65, 5853)

# 21. rob-partial-base-loss-continue
P = 'rob-partial-base-loss-continue'
C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'destroy@1200:agent-region{11,30,r4}',
    'Core base: fact + proc + powr + weap + fix @ y=18; north harv @(10,16); south outpost (pbox + harv) @ y=30 (destroyed at t=1200); 4× medium tank @(22,18-24) stance:2; starting_cash $2000',
    'Lane-mouth pickets: 2× 1tnk @(24,19/23) stance:2. Eastern garrison: 5× rifle @(80,18-24 + 82,20) defending enemy fact, turtle bot. Sentinel fact @(120,20)',
    "Commander, GRACEFUL DEGRADATION. Four medium tanks staged at (22, 18..24) in front of a full core base (yard, refinery, power, war factory, depot). A south outpost holds an outer pillbox and one of two harvesters. Around tick 1200 a strike RAZES the south outpost — pillbox and harvester gone, core untouched. Don't rebuild the lost outpost. Build ONE more medium tank, then attack-move east to clear five rifles guarding the enemy yard. Five tanks alive, five kills, yard standing, within about 60 turns.",
    '≥5 medium tanks alive AND ≥5 kills AND own yard alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'destroy@1200:agent-region',
    'Same core base; harder non-critical loss (south harv + second outer pbox); starting_cash $2000',
    'Lane pickets + eastern garrison (slightly heavier than easy); sentinel fact @(120,20)',
    "Commander, same graceful-degradation triage with a harder non-critical loss. The destroy event razes the south harvester AND a second outer pillbox. Core base untouched. Press on — build one more medium tank, attack east to clear the eastern garrison. Five tanks alive, five kills, yard standing, within about 60 turns.",
    '≥5 medium tanks alive AND ≥5 kills AND own yard alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'destroy@1200:agent-region',
    'Same core base; heaviest non-critical loss; starting_cash $2000',
    'Heavier picket and eastern garrison; sentinel fact @(120,20). Establishment bar tightens to 6 tanks',
    "Commander, same graceful-degradation triage on the highest establishment bar — six tanks instead of five. Around tick 1200 a strike razes the south outpost. Don't rebuild the lost outpost — build TWO more medium tanks and attack east to clear the eastern garrison. Six tanks alive, five kills, yard standing, within about 60 turns.",
    '≥6 medium tanks alive AND ≥5 kills AND own yard alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    60, 5403)

# 22. rob-unexpected-enemy-spawn
P = 'rob-unexpected-enemy-spawn'
C = 'reasoning'
T = 'observe, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'fact @(10,20) + proc @(6,20) + 5 defenders (3× 2tnk + 2× e3) stance:2 @ (14-16, 18-22); starting_cash $0; no tent (no replacements)',
    'Wave 1 (visible): 4× rifle infantry @(80,20) stance:3 hunt — charges centroid. Wave 2 (HIDDEN at fog corner): 3× rifle infantry @(90,8) stance:3 hunt — closes by ~tick 1500. Sentinel fact @(120,20)',
    "Commander, RESERVE-FOR-FOLLOW-ON. An obvious 4-rifle cluster centred on the arena charges your base; a SECOND 3-rifle wave hidden at a fog corner closes from an unexpected bearing around tick 1500. Your five defenders (three tanks plus two rocket soldiers) are your ENTIRE force — no barracks, no cash. Actively concentrate fire and intercept both waves. Seven kills, yard alive, two units alive, within about 60 turns.",
    '≥7 kills AND yard alive AND ≥2 own units alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    90, 8103)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'Same defenders, no cash, no tent',
    'Wave 1 unchanged. Wave 2: 4× rifle infantry @(90,8) hunt (equal size). Kill bar lifts to 8',
    "Commander, same surprise-second-wave doctrine, but Wave 2 is now equal in size to Wave 1 (four rifles each). Same five defenders, no reinforcements. Eight kills, yard alive, two units alive, within about 60 turns.",
    '≥8 kills AND yard alive AND ≥2 own units alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    90, 8103)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'Same defenders, no cash, no tent',
    'Wave 1: 4× rifle hunt. Wave 2: 5× rifle hidden at fog corner, hunt. Kill bar lifts to 9',
    "Commander, same doctrine with Wave 2 LARGER than Wave 1 (five rifles vs four). Same five defenders, no reinforcements. Nine kills, yard alive, two units alive, within about 60 turns.",
    '≥9 kills AND yard alive AND ≥2 own units alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    90, 8103)

# 23. rob-unit-loss-recovery
P = 'rob-unit-loss-recovery'
C = 'reasoning'
T = 'observe, build, place_building, move_units, attack_unit, attack_move, stop'
add(P, 'easy', C, M, SZ, 'wide', T, '[]', 'none',
    'Soviet base: fact + powr + weap + fix; 4× heavy tank (3tnk) staged @(22,18-24) stance:2; starting_cash $1500 (exactly one 3tnk)',
    'Lane pickets: 2× 4tnk @(24,19/23) stance:3. Eastern garrison: 5× rifle @(80,18-24+82,20) stance:1, turtle bot. Sentinel fact @(120,20)',
    "Commander, FORCE-REGEN UNDER ATTRITION. Four heavy tanks staged in front of a production base (yard, power, war factory, depot). Two enemy mammoth tanks picket the lane mouth at x=24; five rifles guard the enemy yard at x=80. Your column is BELOW the five-tank establishment — your $1500 reserve covers exactly one more heavy tank. Build it, attack east. Five tanks alive, five kills, yard standing, within about 60 turns.",
    '≥5 heavy tanks alive AND ≥5 kills AND own yard alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'medium', C, M, SZ, 'wide', T, '[]', 'none',
    'Same Soviet base; 4× 3tnk staged + $1500 reserve',
    'Three lane-mouth heavies (+1 vs easy); eastern garrison 6× rifle (+1)',
    "Commander, same force-regen task with heavier pressure. THREE enemy mammoth tanks now picket the lane mouth — a stalling column is ground down fast. The eastern garrison is six rifles. Same five-tank establishment, same $1500 reserve. Within about 60 turns.",
    '≥5 heavy tanks alive AND ≥5 kills AND own yard alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    60, 5403)
add(P, 'hard', C, M, SZ, 'wide', T, '[]', 'none',
    'Same Soviet base; $3000 reserve (exactly two 3tnk)',
    'Heavier lane pickets and garrison; establishment lifts to 6 tanks',
    "Commander, same force-regen with the establishment bar lifted to SIX heavy tanks. Your $3000 reserve covers exactly two more 3tnks. Build them, attack east, six tanks alive, five kills, yard standing, within about 60 turns.",
    '≥6 heavy tanks alive AND ≥5 kills AND own yard alive, within 5400 ticks.',
    'No units, yard destroyed, or deadline (5401 ticks).',
    60, 5403)


# ──────────────────────────────────────────────────────────────────
# Enemy posture overlay (F1 §8). Family-7 packs use few combat bots;
# many have NO enemy combatant (the procedural-compliance signal is
# tool / order discipline, not battle).
# ──────────────────────────────────────────────────────────────────
POSTURE = {
    'proc-checklist-no-deviation':            ('NONE (no enemy on map)', ''),
    'proc-conditional-branch-action':         ('guard (outpost cluster holds post, leashed lunge)', ''),
    'proc-instruction-following-edge-case':   ('NONE (no enemy on map)', ''),
    'proc-no-attack-passive-only':            ('static (gun turret sentry posts, st3 AttackAnything; turrets cannot pursue)', ''),
    'proc-only-build-no-combat':              ('passive (st0 HoldFire infantry, easy/medium) / hunt (chipping squad, hard) — intended to NOT raze the yard inside the budget', ''),
    'proc-only-defend-no-attack':             ('patrol (probe oscillates onto defender footprint); st2 probes', ''),
    'proc-ordered-action-strict':             ('NONE (no enemy on map; pre-placed sentinel fact only)', ''),
    'proc-strict-toolban-fidelity':           ('NONE (no enemy combatant — quiet baseline; sentinel fact only)', ''),
    'proc-tool-use-multi-distractor':         ('passive (st0 off-path garrison + st0 central sentry stack)', 'NOTE: hard tier briefing says central sentry stack is "stance:3 fire arcs" but YAML actually sets stance:0 — the central corridor is NOT denied; cosmetic mismatch since spawn lanes route around y=20 anyway'),
    'proc-tool-use-with-distractor':          ('passive (st0 far-off sentry, easy) / patrol (medium/hard) — light pressure, distractor not threat', ''),
    'strict-production-bom':                  ('static (st2 lone enemy rifle far off-map)', ''),
    'strict-sequence':                        ('NONE (no enemy on map)', ''),
    'strict-toolban-fidelity-under-pressure': ('patrol (probe oscillates central corridor; salient temptation only)', ''),
    'maint-repair-priority-order':            ('scripted strike (scheduled_events spawn_actors at t=720, destroy_actors at t=820 — st3 grenadiers, bounded burst)', ''),
    'maint-sell-and-recoup-cash':             ('NONE (sentinel fact only; pure budget puzzle)', ''),
    'rob-cash-depletion-recovery':            ('hunt (4tnk strike on the proc; pre-placed at t=0, NOT scheduled)', ''),
    'rob-deadline-shortened-midway':          ('static (target fact is the only enemy actor)', ''),
    'rob-multiple-simultaneous-pressures':    ('hunt (base-charge rifle squad + patch-raid 1tnk; both via hunt bot)', ''),
    'rob-objective-change-midway':            ('static (st2 region pickets and yard guard)', ''),
    'rob-objective-shift-with-or-clause':     ('static (st2 yard A rocket cluster and yard B picket)', ''),
    'rob-partial-base-loss-continue':         ('turtle (eastern garrison holds; lane-mouth pickets st2)', ''),
    'rob-unexpected-enemy-spawn':             ('hunt (both waves st3 + hunt bot; Wave 2 pre-placed at fog corner, NOT scheduled)', ''),
    'rob-unit-loss-recovery':                 ('turtle (eastern e1 garrison) + st3 4tnk lane pickets', ''),
}
for r in R:
    p, pi = POSTURE.get(r['pack'], ('', 'UNCATEGORISED'))
    r['enemy_posture'] = p
    r['posture_issue'] = pi


# ──────────────────────────────────────────────────────────────────
# Emit CSV
# ──────────────────────────────────────────────────────────────────
fields = ['pack', 'level', 'capability', 'map_name', 'map_size',
          'map_fit', 'tools', 'forbidden_tools', 'scheduled_events',
          'agent_force', 'enemy_force', 'enemy_posture',
          'posture_issue', 'briefing_RA', 'win_condition',
          'lose_condition', 'max_turns', 'tick_budget']
with OUT.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
    w.writeheader()
    for r in R:
        w.writerow(r)
print(f'Wrote {len(R)} rows to {OUT}')
